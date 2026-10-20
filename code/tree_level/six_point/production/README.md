# Sphere 1→5 — production

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../../reproduce.py) and the [repository README](../../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [dispatch_shared_refinement_queue.py](dispatch_shared_refinement_queue.py) | Checking saved inputs and distributing refinement jobs |
| [six_point_prepare_amplitude_curve.py](six_point_prepare_amplitude_curve.py) | Preparing the amplitude calculation and running its stages |
| [six_point_refinement.py](six_point_refinement.py) | Selective refinement through $N=14$ and continuation through $N=18$ |
| [six_point_run_frozen_energy_worker.py](six_point_run_frozen_energy_worker.py) | Preparing the amplitude calculation and running its stages |
| [six_point_run_frozen_radial_worker.py](six_point_run_frozen_radial_worker.py) | Preparing the amplitude calculation and running its stages |
| [six_point_submit_radial_campaign.py](six_point_submit_radial_campaign.py) | Preparing the amplitude calculation and running its stages |
| [verify_saved_c14_continuation_inputs.py](verify_saved_c14_continuation_inputs.py) | Checking saved inputs and distributing refinement jobs |
