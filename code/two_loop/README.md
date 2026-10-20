# Genus-two free energy and channel comparisons

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../reproduce.py) and the [repository README](../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [conformal_block/ccy_genus2_glasses_block.py](conformal_block/ccy_genus2_glasses_block.py) | Genus-two conformal block in the glasses channel |
| [geometry/audit_q_to_omega_accuracy.py](geometry/audit_q_to_omega_accuracy.py) | Checks of the period map and modular transformations |
| [geometry/bolza_ccy_recursion.py](geometry/bolza_ccy_recursion.py) | Checks of the period map and modular transformations |
| [geometry/bolza_torus_plumbing_reach.py](geometry/bolza_torus_plumbing_reach.py) | Plumbing atlas and choice of the homology marking |
| [geometry/genus2_calibrated_schottky.py](geometry/genus2_calibrated_schottky.py) | Period map and choice of its numerical implementation |
| [geometry/genus2_holomorphic_period_table.py](geometry/genus2_holomorphic_period_table.py) | Period-map tables and their construction |
| [geometry/genus2_hybrid_period_map.py](geometry/genus2_hybrid_period_map.py) | Period map and choice of its numerical implementation |
| [geometry/genus2_integration_geometry.py](geometry/genus2_integration_geometry.py) | Channel selection and reconstruction of saved plumbing coordinates |
| [geometry/genus2_multiprecision_collocation.py](geometry/genus2_multiprecision_collocation.py) | Period map and choice of its numerical implementation |
| [geometry/genus2_period_table.py](geometry/genus2_period_table.py) | Period-map tables and their construction |
| [geometry/genus2_period_table_grid.py](geometry/genus2_period_table_grid.py) | Period-map tables and their construction |
| [geometry/genus2_period_table_selector.py](geometry/genus2_period_table_selector.py) | Channel selection and reconstruction of saved plumbing coordinates |
| [geometry/genus2_plumbing_atlas.py](geometry/genus2_plumbing_atlas.py) | Plumbing atlas and choice of the homology marking |
| [geometry/liouville_genus2_modular_check.py](geometry/liouville_genus2_modular_check.py) | Checks of the period map and modular transformations |
| [liouville_integration/genus2_liouville_evaluation.py](liouville_integration/genus2_liouville_evaluation.py) | Channel evaluation and logarithmic scaling of the Liouville partition function |
| [liouville_integration/liouville_genus2.py](liouville_integration/liouville_genus2.py) | Separating approximation in the pair-of-tori channel |
| [liouville_integration/liouville_genus2_ccy.py](liouville_integration/liouville_genus2_ccy.py) | Liouville momentum integrals in the theta and glasses channels |
| [liouville_integration/liouville_genus2_glasses.py](liouville_integration/liouville_genus2_glasses.py) | Liouville momentum integrals in the theta and glasses channels |
| [moduli_integration/genus2_bc_ghost.py](moduli_integration/genus2_bc_ghost.py) | Ghost and matter factors and the complete moduli integrand |
| [moduli_integration/genus2_c1_string_integrand.py](moduli_integration/genus2_c1_string_integrand.py) | Ghost and matter factors and the complete moduli integrand |
| [moduli_integration/genus2_moduli_estimation.py](moduli_integration/genus2_moduli_estimation.py) | Sampling weights, moduli estimates, and their statistical errors |
| [moduli_integration/genus2_moduli_sampling.py](moduli_integration/genus2_moduli_sampling.py) | Siegel--Gottschling fundamental domain and Monte Carlo sampling |
| [moduli_integration/genus2_node_evaluation.py](moduli_integration/genus2_node_evaluation.py) | Ghost and matter factors and the complete moduli integrand |
| [moduli_integration/genus2_scalar_evaluation.py](moduli_integration/genus2_scalar_evaluation.py) | Scalar partition function and its truncation checks |
| [moduli_integration/genus2_separating_tail.py](moduli_integration/genus2_separating_tail.py) | Separating approximation, radial integral, and sampled remainder |
| [moduli_integration/genus2_siegel_fundamental_domain.py](moduli_integration/genus2_siegel_fundamental_domain.py) | Siegel--Gottschling fundamental domain and Monte Carlo sampling |
| [other_scripts/assemble_genus2_resolved_production.py](other_scripts/assemble_genus2_resolved_production.py) | Combining bulk and separating contributions |
| [other_scripts/assemble_genus2_tail_completion.py](other_scripts/assemble_genus2_tail_completion.py) | Combining bulk and separating contributions |
| [other_scripts/audit_genus2_g_convention.py](other_scripts/audit_genus2_g_convention.py) | Radius dependence, figures, and checks of normalization conventions |
| [other_scripts/checks/ccy_genus2_glasses_block_checks.py](other_scripts/checks/ccy_genus2_glasses_block_checks.py) | Independent checks of the glasses conformal block |
| [other_scripts/checks/genus2_c1_string_integrand_checks.py](other_scripts/checks/genus2_c1_string_integrand_checks.py) | Checks of the moduli integrand |
| [other_scripts/checks/genus2_separating_tail_checks.py](other_scripts/checks/genus2_separating_tail_checks.py) | Checks of the separating approximation, precision, and tail assembly |
| [other_scripts/checks/genus2_tail_completion_checks.py](other_scripts/checks/genus2_tail_completion_checks.py) | Checks of the separating approximation, precision, and tail assembly |
| [other_scripts/checks/genus2_tail_precision_checks.py](other_scripts/checks/genus2_tail_precision_checks.py) | Checks of the separating approximation, precision, and tail assembly |
| [other_scripts/checks/plot_genus2_absolute_bry_checks.py](other_scripts/checks/plot_genus2_absolute_bry_checks.py) | Checks of the plotted free energy |
| [other_scripts/genus2_glasses_production_gate.py](other_scripts/genus2_glasses_production_gate.py) | Bulk production and checks of numerical convergence |
| [other_scripts/genus2_moduli_production.py](other_scripts/genus2_moduli_production.py) | Bulk production and checks of numerical convergence |
| [other_scripts/genus2_separating_production.py](other_scripts/genus2_separating_production.py) | Separating-region production and refinement of saved nodes |
| [other_scripts/genus2_separating_residual_retry.py](other_scripts/genus2_separating_residual_retry.py) | Separating-region production and refinement of saved nodes |
| [other_scripts/plot_genus2_absolute_bry.py](other_scripts/plot_genus2_absolute_bry.py) | Radius dependence, figures, and checks of normalization conventions |
| [other_scripts/rebuild_genus2_bulk_control_groups.py](other_scripts/rebuild_genus2_bulk_control_groups.py) | Reconstructing bulk control groups and the separating correction |
| [other_scripts/recompute_channel_comparison.py](other_scripts/recompute_channel_comparison.py) | Recompute theta/glasses Q_L at the five printed manuscript period matrices. |
| [other_scripts/study_genus2_separating_tail_correction.py](other_scripts/study_genus2_separating_tail_correction.py) | Reconstructing bulk control groups and the separating correction |
