# Sphere 1→5 — liouville integration

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../../reproduce.py) and the [repository README](../../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [six_point_analyze_quadrature_and_order_controls.py](six_point_analyze_quadrature_and_order_controls.py) | Assessment of momentum integration and block truncation |
| [six_point_build_momentum_banks.py](six_point_build_momentum_banks.py) | Building coefficient tables and identifying repeated momentum nodes |
| [six_point_collar_momentum_quadrature.py](six_point_collar_momentum_quadrature.py) | Momentum integration in $D_{\mathrm{sep}}$ and analytic radial integrals |
| [six_point_compare_momentum_quadratures.py](six_point_compare_momentum_quadratures.py) | Assessment of momentum integration and block truncation |
| [six_point_deduplicate_momentum_nodes.py](six_point_deduplicate_momentum_nodes.py) | Building coefficient tables and identifying repeated momentum nodes |
| [six_point_endpoint_and_tail_quadrature.py](six_point_endpoint_and_tail_quadrature.py) | Momentum integration grids |
| [six_point_mixed_order_node_catalog.py](six_point_mixed_order_node_catalog.py) | Building coefficient tables and identifying repeated momentum nodes |
| [six_point_momentum_banks_and_reweighting.py](six_point_momentum_banks_and_reweighting.py) | Liouville momentum integration and stored coefficient tables |
| [six_point_momentum_control_proposals.py](six_point_momentum_control_proposals.py) | Momentum integration grids |
| [six_point_reuse_saved_coefficients.py](six_point_reuse_saved_coefficients.py) | Reusing previously computed coefficients |
