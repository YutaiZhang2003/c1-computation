# Genus-one 1→2 — moduli integration

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../../reproduce.py) and the [repository README](../../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [genus1_analytic_boundary_layer.py](genus1_analytic_boundary_layer.py) | Partition of moduli space, sampling weights, and collision-region maps |
| [genus1_boundary_batch.py](genus1_boundary_batch.py) | Reuse of moduli-dependent factors |
| [genus1_boundary_cubature.py](genus1_boundary_cubature.py) | Deterministic quadrature and assembly of the moduli integral |
| [genus1_boundary_integration.py](genus1_boundary_integration.py) | Deterministic quadrature and assembly of the moduli integral |
| [genus1_collision_sampling.py](genus1_collision_sampling.py) | Partition of moduli space, sampling weights, and collision-region maps |
| [genus1_comb_collar.py](genus1_comb_collar.py) | Collision subtractions and integrated degeneration terms |
| [genus1_cusp_control.py](genus1_cusp_control.py) | Nonseparating approximation and tube integrals |
| [genus1_integrated_scale_collar.py](genus1_integrated_scale_collar.py) | Collision subtractions and integrated degeneration terms |
| [genus1_modular_geometry.py](genus1_modular_geometry.py) | Plumbing parameters, channel choice, and sewing domains |
| [genus1_moduli_panels.py](genus1_moduli_panels.py) | Deterministic quadrature and assembly of the moduli integral |
| [genus1_moduli_sampling.py](genus1_moduli_sampling.py) | Partition of moduli space, sampling weights, and collision-region maps |
| [genus1_nonseparating.py](genus1_nonseparating.py) | Nonseparating approximation and tube integrals |
| [genus1_pair_collar.py](genus1_pair_collar.py) | Collision subtractions and integrated degeneration terms |
| [genus1_sewing_domains.py](genus1_sewing_domains.py) | Plumbing parameters, channel choice, and sewing domains |
| [genus1_three_point_channel_atlas.py](genus1_three_point_channel_atlas.py) | Plumbing parameters, channel choice, and sewing domains |
| [genus1_three_point_collision.py](genus1_three_point_collision.py) | Collision subtractions and integrated degeneration terms |
| [thermal/kernels.py](thermal/kernels.py) | Thermal kinematics and integrated boundary kernels |
| [thermal/radius_scan.py](thermal/radius_scan.py) | Moduli sampling for the thermal correction |
| [thermal/remainders.py](thermal/remainders.py) | Thermal remainder terms |
| [thermal/study_pair_sampling.py](thermal/study_pair_sampling.py) | Moduli sampling for the thermal correction |
