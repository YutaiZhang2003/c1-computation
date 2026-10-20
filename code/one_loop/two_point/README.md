# Genus-one 1→1

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../reproduce.py) and the [repository README](../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [conformal_block/torus_two_point_blocks.py](conformal_block/torus_two_point_blocks.py) | Necklace and OPE conformal blocks, coefficient transformations, and direct descendant checks |
| [liouville_integration/genus1_two_point_coefficient_bank.py](liouville_integration/genus1_two_point_coefficient_bank.py) | Preparation and reuse of the two-point Liouville coefficient banks |
| [liouville_integration/genus1_two_point_worldsheet.py](liouville_integration/genus1_two_point_worldsheet.py) | DOZZ factors, Liouville momentum quadrature, channel evaluation, and leading collision term |
| [moduli_integration/genus1_two_point_thermal_integration.py](moduli_integration/genus1_two_point_thermal_integration.py) | Thermal moduli integration over the full cusp and collision disc, with paired noncompact estimates |
| [moduli_integration/integrate_genus1_two_point_worldsheet.py](moduli_integration/integrate_genus1_two_point_worldsheet.py) | Noncompact moduli integration and cusp checks |
| [other_scripts/checks/genus1_two_point_worldsheet_checks.py](other_scripts/checks/genus1_two_point_worldsheet_checks.py) | Checks of the integrand, channel overlap, and modular covariance |
| [other_scripts/checks/torus_necklace_blocks_checks.py](other_scripts/checks/torus_necklace_blocks_checks.py) | Independent checks of the necklace h-recursion |
| [other_scripts/checks/torus_two_point_blocks_checks.py](other_scripts/checks/torus_two_point_blocks_checks.py) | Independent block checks against direct descendants |
| [other_scripts/fit_genus1_two_point_bry_scan.py](other_scripts/fit_genus1_two_point_bry_scan.py) | Fits and convention checks for saved noncompact amplitudes |
| [other_scripts/genus1_two_point_thermal_production.py](other_scripts/genus1_two_point_thermal_production.py) | Thermal radius scan using the saved production settings |
| [other_scripts/genus1_two_point_thermal_report.py](other_scripts/genus1_two_point_thermal_report.py) | Assembly of the thermal amplitude, sampling errors, and radius figure |
| [other_scripts/plot_genus1_two_point_hybrid_recursion.py](other_scripts/plot_genus1_two_point_hybrid_recursion.py) | Noncompact amplitude figure and saved recursion comparisons |
| [other_scripts/run_genus1_two_point_imaginary_scan.py](other_scripts/run_genus1_two_point_imaginary_scan.py) | Noncompact amplitude scan used for comparison |
