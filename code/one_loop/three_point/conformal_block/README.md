# Genus-one 1→2 — conformal block

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../../reproduce.py) and the [repository README](../../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [genus1_collision_elliptic.py](genus1_collision_elliptic.py) | Elliptic blocks for collision limits and precomputed coefficients |
| [genus1_native_log_evaluation.py](genus1_native_log_evaluation.py) | Evaluation of the retained native series without overflow |
| [torus_c_recursion_numerics.py](torus_c_recursion_numerics.py) | c-recursion and precision checks |
| [torus_three_point_blocks.py](torus_three_point_blocks.py) | Torus three-point blocks in the necklace, pair OPE, and comb channels |
