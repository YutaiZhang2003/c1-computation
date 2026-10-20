# Sphere 1→5 — conformal block

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../../reproduce.py) and the [repository README](../../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [six_point_arbitrary_precision_c_recursion.py](six_point_arbitrary_precision_c_recursion.py) | Six-point block from the $c$-recursion and checks of arithmetic precision |
| [six_point_block_coefficient_provider.py](six_point_block_coefficient_provider.py) | Six-point block from the $c$-recursion and checks of arithmetic precision |
| [six_point_collar_block_evaluator.py](six_point_collar_block_evaluator.py) | Four-point elliptic block and outer descendant corrections in $D_{\mathrm{sep}}$ |
| [six_point_measure_block_convergence.py](six_point_measure_block_convergence.py) | Assessment of block truncation |
| [six_point_refinement_statistics.py](six_point_refinement_statistics.py) | Refinement criteria and paired Monte Carlo statistics |
| [six_point_select_integration_channels.py](six_point_select_integration_channels.py) | Channel selection |
