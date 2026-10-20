# c=1 string computations

Numerical code and lightweight data for the worldsheet calculations in the
`c1_higher_genus_new_edit` draft. The code computes sphere and torus scattering
amplitudes, their thermal counterparts, and the genus-two thermal free energy.

This guide follows **Appendix B: Numerical Details**. For each calculation it
identifies the conformal-block construction, Liouville momentum integral,
moduli integral and numerical checks. File roles follow the sector
`CODE_INDEX.json` inventories. Detailed commands and the complete file catalogue
are in [CODE_REFERENCE.md](CODE_REFERENCE.md).

## From Appendix B to the code

| Draft section | Calculation | Source directory | JSON code inventory |
| --- | --- | --- | --- |
| B.1 | Genus-zero 1→4 amplitude: five-point sphere correlator | [`code/tree_level/five_point/`](code/tree_level/five_point/) | [CODE_INDEX.json](code/tree_level/five_point/CODE_INDEX.json) |
| B.2 | Genus-zero 1→5 amplitude: six-point sphere correlator | [`code/tree_level/six_point/`](code/tree_level/six_point/) | [CODE_INDEX.json](code/tree_level/six_point/CODE_INDEX.json) |
| B.3 | Genus-one 1→1 amplitude: two-point torus correlator | [`code/one_loop/two_point/`](code/one_loop/two_point/) | [CODE_INDEX.json](code/one_loop/two_point/CODE_INDEX.json) |
| B.4 | Genus-one 1→2 amplitude: three-point torus correlator | [`code/one_loop/three_point/`](code/one_loop/three_point/) | [CODE_INDEX.json](code/one_loop/three_point/CODE_INDEX.json) |
| B.5 | Genus-two thermal free energy | [`code/two_loop/`](code/two_loop/) | [CODE_INDEX.json](code/two_loop/CODE_INDEX.json) |

Each inventory gives a source file's `path`, computational `group`, `purpose`
and `sha256`. The groups correspond to the stages in Appendix B:

| Code group | Mathematical task | Output passed to the next stage |
| --- | --- | --- |
| `conformal_block/` | Construct Virasoro blocks at Liouville central charge c=25, using c-recursion and the appropriate degeneration expansions | Block coefficients or evaluated blocks in a chosen sewing channel |
| `liouville_integration/` | Combine blocks with DOZZ structure constants and integrate internal Liouville momenta | A Liouville correlator or partition function at fixed moduli |
| `moduli_integration/` | Include coordinate Jacobians, matter/ghost factors, sampling weights and boundary subtractions | Weighted moduli contributions and integral estimates |
| `geometry/` (genus two) | Relate plumbing coordinates to a marked period matrix and select an admissible chart | Plumbing parameters and period-map diagnostics |
| `production/` and `other_scripts/` | Prepare inputs, run the stages, combine estimates and assess numerical stability | Amplitudes, free energies, uncertainty records and figures |

The functions in these groups implement the numerical calculation. The
`recover_*` programs reconstruct the reported observables from saved integrated
inputs. Matrix-model formulas enter the final comparisons; they do not generate
the worldsheet integration data.

## B.1 Sphere five-point amplitude

The five-point code computes the genus-zero 1→4 amplitude. It enumerates the
120 labelled decomposition channels, maps the punctures to
`(0, x, y, 1, infinity)`, and uses plumbing coordinates `(q1, q2) = (x/y, y)`.
The selected channel minimizes `max(|q1|, |q2|)` among admissible frames.

| Appendix B step | Main code | What it implements |
| --- | --- | --- |
| Conformal block and channel choice | [five_point_liouville_correlator_and_atlas.py](code/tree_level/five_point/conformal_block/five_point_liouville_correlator_and_atlas.py), [five_point_virasoro_block_recursion.py](code/tree_level/five_point/conformal_block/five_point_virasoro_block_recursion.py) | Coordinate frames, channel selection and the five-point c-recursion, truncated by total degree in q1 and q2. |
| Liouville momentum integration | [five_point_integrand_and_quadrature.py](code/tree_level/five_point/liouville_integration/five_point_integrand_and_quadrature.py) | Two-momentum quadrature, reusable DOZZ factors and block coefficients, and crossed-pole residue terms. |
| Moduli integration | [five_point_moduli_integration.py](code/tree_level/five_point/moduli_integration/five_point_moduli_integration.py) | Scrambled Sobol sampling, the chart-mixture density rho_Gamma, coordinate Jacobians, the integral mean and its sampling standard error. |
| Production and assembly | [five_point_production.py](code/tree_level/five_point/other_scripts/five_point_production.py) | Energy scans and the retained scan extension, using the preceding numerical modules. |
| Numerical stability | [ccy_sphere_five_point_checks.py](code/tree_level/five_point/other_scripts/ccy_sphere_five_point_checks.py), [sphere_five_point_continuation_checks.py](code/tree_level/five_point/other_scripts/sphere_five_point_continuation_checks.py), [sphere_five_point_30point_audit_summary.py](code/tree_level/five_point/other_scripts/sphere_five_point_30point_audit_summary.py) | Independent block checks, contour-residue checks and comparisons with saved higher-order evaluations. |

Appendix B.1 uses a finite momentum interval, `P = Pmax*u^(5/4)`, with
`Pmax = 6`. It averages rules with interchanged unequal node counts to avoid
coincident internal momenta. For equal outgoing energies `omega_j = i*t`,
the continuation above `t = 2/5` adds the crossed-pole residue. The moduli
estimator divides by the full chart-mixture density so that overlapping
coordinate frames do not double count the same surface.

**Result:** the sphere 1→4 energy curve and, after the thermal kinematic
substitution, its radius curve. Recovery writes them under
`output/recovered/tree_level/5_point/`.

## B.2 Sphere six-point amplitude

The six-point code computes the genus-zero 1→5 amplitude using 720 linear
channels and 720 star channels. The admissibility test accounts for the
singularities of the block expansion in each topology. Channel selection and
sampling are separate: the integrand can be evaluated in a better-converging
channel while retaining the original sampling measure.

| Appendix B step | Main code | What it implements |
| --- | --- | --- |
| Conformal block | [six_point_arbitrary_precision_c_recursion.py](code/tree_level/six_point/conformal_block/six_point_arbitrary_precision_c_recursion.py), [six_point_block_coefficient_provider.py](code/tree_level/six_point/conformal_block/six_point_block_coefficient_provider.py) | Six-point c-recursion and verified coefficient production, with arithmetic-precision checks. |
| Channel coordinates and selection | [six_point_channel_coordinates.py](code/tree_level/six_point/moduli_integration/six_point_channel_coordinates.py), [six_point_select_integration_channels.py](code/tree_level/six_point/conformal_block/six_point_select_integration_channels.py) | Linear/star frames, coordinate transformations, overlap densities and channel choice. |
| Liouville momentum integration | [six_point_endpoint_and_tail_quadrature.py](code/tree_level/six_point/liouville_integration/six_point_endpoint_and_tail_quadrature.py), [six_point_momentum_banks_and_reweighting.py](code/tree_level/six_point/liouville_integration/six_point_momentum_banks_and_reweighting.py), [six_point_build_momentum_banks.py](code/tree_level/six_point/liouville_integration/six_point_build_momentum_banks.py) | Endpoint and infinite-tail rules, stored CFT coefficient banks and their reuse in weighted momentum integrals. |
| Moduli integration | [six_point_radial_sampling_distribution.py](code/tree_level/six_point/moduli_integration/six_point_radial_sampling_distribution.py), [six_point_sample_integration_geometry.py](code/tree_level/six_point/moduli_integration/six_point_sample_integration_geometry.py), [six_point_integrate_mixed_order_amplitudes.py](code/tree_level/six_point/moduli_integration/six_point_integrate_mixed_order_amplitudes.py) | Radial-mixture sampling, the weighted integrand W_Gamma and integration with explicit block truncations. |
| Separating region D_sep | [six_point_collar_block_evaluator.py](code/tree_level/six_point/conformal_block/six_point_collar_block_evaluator.py), [six_point_analytic_collar_subtraction.py](code/tree_level/six_point/moduli_integration/six_point_analytic_collar_subtraction.py), [six_point_collar_momentum_quadrature.py](code/tree_level/six_point/liouville_integration/six_point_collar_momentum_quadrature.py) | The four-point elliptic block in the two-long-tube limit, its subtraction from the sampled integrand and its integrated contribution. |
| Numerical stability and assembly | [six_point_measure_block_convergence.py](code/tree_level/six_point/conformal_block/six_point_measure_block_convergence.py), [six_point_refinement.py](code/tree_level/six_point/production/six_point_refinement.py), [six_point_refinement_assembly.py](code/tree_level/six_point/moduli_integration/six_point_refinement_assembly.py), [six_point_assemble_radial_curve_covariance.py](code/tree_level/six_point/moduli_integration/six_point_assemble_radial_curve_covariance.py) | Block-order comparisons, selected higher-order corrections and combination of the independent sampling estimates with their covariance. |

Each internal momentum is split into a finite interval and an infinite tail.
Their scales depend on the plumbing coordinates and external energy. The
moduli proposal mixes samples near degeneration limits with bulk samples.
In `D_sep`, the two outer tubes are approximated by primary propagation, leaving
a four-point block in the middle; the outer radial integrals are performed
analytically. Subtracting the same approximation from the sampled contribution
allows the integrated term to be added without double counting.

**Result:** the direct-series sphere 1→5 curve, its thermal radius curve and
cross-energy covariance, under `output/recovered/tree_level/6_point/`.

## B.3 Torus two-point amplitude

The torus two-point code computes the genus-one 1→1 amplitude. It uses the
necklace channel away from a collision and the OPE channel when the two
operators approach one another. In Appendix B.3 the necklace variables satisfy
`q1*q2 = exp(2*pi*i*tau)`; the OPE expansion uses the handle nome and
`v = exp(-i*z) - 1`.

| Appendix B step | Main code | What it implements |
| --- | --- | --- |
| Conformal block | [torus_two_point_blocks.py](code/one_loop/two_point/conformal_block/torus_two_point_blocks.py) | Necklace and OPE blocks, coefficient transformations and direct descendant comparisons. |
| Liouville momentum integration | [genus1_two_point_worldsheet.py](code/one_loop/two_point/liouville_integration/genus1_two_point_worldsheet.py), [genus1_two_point_coefficient_bank.py](code/one_loop/two_point/liouville_integration/genus1_two_point_coefficient_bank.py) | DOZZ factors, momentum quadrature, channel evaluation and reusable coefficient banks. |
| Moduli integration | [integrate_genus1_two_point_worldsheet.py](code/one_loop/two_point/moduli_integration/integrate_genus1_two_point_worldsheet.py), [genus1_two_point_thermal_integration.py](code/one_loop/two_point/moduli_integration/genus1_two_point_thermal_integration.py) | Noncompact and thermal integration, including the cusp and collision disc, with paired estimates. |
| Production and assembly | [genus1_two_point_thermal_production.py](code/one_loop/two_point/other_scripts/genus1_two_point_thermal_production.py), [genus1_two_point_thermal_report.py](code/one_loop/two_point/other_scripts/genus1_two_point_thermal_report.py) | Prepare coefficient banks, run the radius scan and assemble amplitudes and sampling errors. |
| Numerical stability | [torus_two_point_blocks_checks.py](code/one_loop/two_point/other_scripts/checks/torus_two_point_blocks_checks.py), [genus1_two_point_worldsheet_checks.py](code/one_loop/two_point/other_scripts/checks/genus1_two_point_worldsheet_checks.py) | Independent block checks, channel overlap and modular-covariance checks. |

The momentum rule uses `P = 6*u^2` to resolve small momenta. The moduli
integration maps the torus variables to a unit hypercube. In the collision
disc, the primary OPE contribution is integrated analytically and subtracted
from the numerical remainder. The thermal implementation includes the compact
boson factor while retaining paired noncompact estimates.

The baseline settings, including block orders, momentum nodes and sampling
parameters, are recorded in
[`production_settings.json`](code/one_loop/two_point/production_settings.json).
Saved comparison receipts record variations of those settings.

**Result:** the thermal 1→1 radius curve, noncompact checks and native-q
convergence tables, under `output/recovered/one_loop/two_point/`.

## B.4 Torus three-point amplitude

The torus three-point code computes the genus-one 1→2 amplitude in the
necklace, pair-OPE and comb channels. Its moduli integration treats collision
and long-tube regions explicitly: primary propagation permits some tube
integrals to be done analytically, and Monte Carlo sampling covers the remaining
compact region.

| Appendix B step | Main code | What it implements |
| --- | --- | --- |
| Conformal block | [torus_three_point_blocks.py](code/one_loop/three_point/conformal_block/torus_three_point_blocks.py), [torus_c_recursion_numerics.py](code/one_loop/three_point/conformal_block/torus_c_recursion_numerics.py), [genus1_collision_elliptic.py](code/one_loop/three_point/conformal_block/genus1_collision_elliptic.py) | Native blocks in the three channels, c-recursion with precision checks and elliptic blocks in degeneration limits. |
| Liouville momentum integration | [genus1_three_point_momentum.py](code/one_loop/three_point/liouville_integration/genus1_three_point_momentum.py), [genus1_three_point_worldsheet.py](code/one_loop/three_point/liouville_integration/genus1_three_point_worldsheet.py), [genus1_boundary_spectrum.py](code/one_loop/three_point/liouville_integration/genus1_boundary_spectrum.py) | Momentum rules, DOZZ/block inputs and correlator evaluation at fixed moduli. |
| Geometry and moduli partition | [genus1_three_point_channel_atlas.py](code/one_loop/three_point/moduli_integration/genus1_three_point_channel_atlas.py), [genus1_analytic_boundary_layer.py](code/one_loop/three_point/moduli_integration/genus1_analytic_boundary_layer.py), [genus1_moduli_sampling.py](code/one_loop/three_point/moduli_integration/genus1_moduli_sampling.py) | Plumbing coordinates, channel selection, the partition into integration regions and sampling weights. |
| Collision and long-tube terms | [genus1_pair_collar.py](code/one_loop/three_point/moduli_integration/genus1_pair_collar.py), [genus1_comb_collar.py](code/one_loop/three_point/moduli_integration/genus1_comb_collar.py), [genus1_nonseparating.py](code/one_loop/three_point/moduli_integration/genus1_nonseparating.py), [genus1_boundary_integration.py](code/one_loop/three_point/moduli_integration/genus1_boundary_integration.py) | Pair/comb subtractions, the nonseparating approximation and numerical integration of the remaining boundary variables. |
| Thermal calculation | [thermal momentum code](code/one_loop/three_point/liouville_integration/thermal/), [thermal moduli code](code/one_loop/three_point/moduli_integration/thermal/), [thermal production code](code/one_loop/three_point/other_scripts/thermal/) | Thermal momentum placement and banks, compact-boson kernels, matched remainder terms and radius-curve assembly. |
| Numerical stability and assembly | [genus1_moduli_stability.py](code/one_loop/three_point/other_scripts/genus1_moduli_stability.py), [genus1_three_point_c_recursion_checks.py](code/one_loop/three_point/other_scripts/checks/genus1_three_point_c_recursion_checks.py), [genus1_amplitude_report.py](code/one_loop/three_point/other_scripts/genus1_amplitude_report.py) | Moduli-quadrature comparisons, independent recursion checks and amplitude/error reports. |

As in the six-point sphere calculation, momentum quadrature separates small
momenta from infinite tails. Pair and triple collisions use their respective
OPE/comb descriptions. In the nonseparating long-tube limit, the remaining
sphere five-point block is evaluated through elliptic recursion. The compact
and degeneration contributions are then assembled with their original
uncertainty definitions.

**Result:** the noncompact 1→2 energy curve and its thermal radius curve, under
`output/recovered/one_loop/three_point/` and
`output/recovered/one_loop/three_point_thermal/`.

## B.5 Genus-two thermal free energy

The genus-two computation starts from a period matrix Omega. It finds theta
or glasses plumbing coordinates for that marked surface, evaluates the
Liouville and free-boson partition functions in the same frame, and constructs
the full moduli integrand for the free energy.

| Appendix B step | Main code | What it implements |
| --- | --- | --- |
| Plumbing parameters and period matrix | [plumbing_algorithms.py](code/general/geometry/plumbing_algorithms.py), [genus2_hybrid_period_map.py](code/two_loop/geometry/genus2_hybrid_period_map.py), [genus2_multiprecision_collocation.py](code/two_loop/geometry/genus2_multiprecision_collocation.py) | Schottky period evaluation and normalized holomorphic-form collocation, including multiprecision evaluation. |
| Chart selection | [genus2_plumbing_atlas.py](code/two_loop/geometry/genus2_plumbing_atlas.py), [genus2_integration_geometry.py](code/two_loop/geometry/genus2_integration_geometry.py) | Search over theta/glasses representations, retain the homology marking and check the reconstructed period matrix. |
| Conformal block | [ccy_genus2_block.py](code/general/virasoro/ccy_genus2_block.py), [ccy_genus2_glasses_block.py](code/two_loop/conformal_block/ccy_genus2_glasses_block.py), [genus2_vacuum_blocks.py](code/general/virasoro/genus2_vacuum_blocks.py) | Theta/glasses c-recursion, the resummed global block and the large-c vacuum contribution. |
| Liouville momentum integration | [liouville_genus2_ccy.py](code/two_loop/liouville_integration/liouville_genus2_ccy.py), [liouville_genus2_glasses.py](code/two_loop/liouville_integration/liouville_genus2_glasses.py), [genus2_liouville_evaluation.py](code/two_loop/liouville_integration/genus2_liouville_evaluation.py) | Three-momentum integrals in the two channels and logarithmic scaling of the partition function. |
| Scalar, ghost and complete integrand | [genus2_scalar_evaluation.py](code/two_loop/moduli_integration/genus2_scalar_evaluation.py), [genus2_bc_ghost.py](code/two_loop/moduli_integration/genus2_bc_ghost.py), [genus2_c1_string_integrand.py](code/two_loop/moduli_integration/genus2_c1_string_integrand.py) | Scalar truncation checks, ghost density and assembly of the c=1 string integrand in period-matrix coordinates. |
| Moduli integration | [genus2_siegel_fundamental_domain.py](code/two_loop/moduli_integration/genus2_siegel_fundamental_domain.py), [genus2_moduli_sampling.py](code/two_loop/moduli_integration/genus2_moduli_sampling.py), [genus2_moduli_estimation.py](code/two_loop/moduli_integration/genus2_moduli_estimation.py) | Gottschling-domain tests, sampling distributions and weights, integral estimates and statistical errors. |
| Separating region | [genus2_separating_tail.py](code/two_loop/moduli_integration/genus2_separating_tail.py), [genus2_separating_production.py](code/two_loop/other_scripts/genus2_separating_production.py), [assemble_genus2_tail_completion.py](code/two_loop/other_scripts/assemble_genus2_tail_completion.py) | Primary propagation through the separating bridge, its analytic radial integral, sampled residuals and assembly. |
| Numerical stability | [audit_q_to_omega_accuracy.py](code/two_loop/geometry/audit_q_to_omega_accuracy.py), [check_channel_convergence.py](code/two_loop/other_scripts/check_channel_convergence.py), [genus2_separating_tail_checks.py](code/two_loop/other_scripts/checks/genus2_separating_tail_checks.py) | Period-map checks, fixed-geometry momentum/block-order comparisons and separating-limit checks. |

The hybrid period map uses Schottky products for small plumbing parameters and
holomorphic-form matching where that representation is better conditioned.
For the blocks, the large-c vacuum product and resummed global contribution
are evaluated separately from the finite-c recursion order.

The three Liouville momentum integrals use the `primary-gaussian` rule in
[`liouville_momentum_quadrature.py`](code/general/numerics/liouville_momentum_quadrature.py).
For each edge, `kappa_e = -2*log(|q_e|)` and `P_e = sqrt(u_e/kappa_e)`;
the u nodes come from generalized Gauss-Laguerre quadrature with weight
`u^(-1/2)*exp(-u)`. Thus the momentum nodes adapt to the tube and cover the
infinite domain without a finite momentum cutoff.

For the moduli integral, the code samples the Siegel-Gottschling domain with
explicit Jacobians and proposal weights. Near the separating limit it
integrates the primary bridge contribution analytically and samples the
remaining correction. Radius-independent Liouville values and geometry are
reused across radii, with correlations retained in the final covariance.

**Result:** the retained genus-two free-energy curve, its correlated fit and
covariance, under `output/recovered/two_loop/`. The reported sampling error
covers the resolved contribution; the residual region
`0 < |Omega12| < 1e-5` and total systematic error remain unbounded.

### Cross-channel matching

The comparison uses

$$
\mathcal Q_L = \frac{Z_L}{Z_X^{25}},
\qquad
\Delta = \frac{\mathcal Q_{L,\mathrm{Theta}}}
                   {\mathcal Q_{L,\mathrm{Glasses}}}-1.
$$

[`recompute_channel_comparison.py`](code/two_loop/other_scripts/recompute_channel_comparison.py)
solves the geometry and evaluates both partition factors.
[`check_channel_convergence.py`](code/two_loop/other_scripts/check_channel_convergence.py)
varies the Liouville momentum and block orders at fixed saved geometry and
scalar normalization.
[`recover_channel_convergence.py`](code/two_loop/other_scripts/recover_channel_convergence.py)
verifies the saved evaluations and reconstructs the current table.

The current table uses **block order 12 and 16 momentum nodes per edge**.
Its five target matrices are the exact decimals in
[`targets.json`](data/channel_comparison/targets.json); the original unrounded
matrix coordinates were unavailable. Both channels represent the same target
surface with their recorded homology markings. The
[full-precision table](data/channel_convergence/current_channel_comparison.csv)
has `max |Delta| = 6.958719225201529e-8`. This is empirical channel agreement
within `1e-7`, rather than an absolute error estimate for either integral.

The saved receipts specify the settings used for each evaluation. In
particular, the current 16-node table supersedes the 12-node comparison still
described in Appendix B.5's prose. The baseline receipts and completed
convergence checks remain available. Values retain full stored binary64
precision; `dps=50` refers to DOZZ arithmetic. Rounded table copies are omitted.

## Shared numerical components

| Component | Code | Role across the calculations |
| --- | --- | --- |
| Liouville structure constants | [liouville_torus.py](code/general/liouville/liouville_torus.py) | Upsilon_b, DOZZ factors and torus one-point momentum integration. |
| Virasoro algebra and recursion | [virasoro_descendant_algebra.py](code/general/virasoro/virasoro_descendant_algebra.py), [virasoro_blocks.py](code/general/virasoro/virasoro_blocks.py), [virasoro_plumbing_graph.py](code/general/virasoro/virasoro_plumbing_graph.py) | Descendant algebra, torus blocks and direct sewing checks. |
| Elliptic degeneration blocks | [sphere_four_point_c25.py](code/general/virasoro/sphere_four_point_c25.py), [genus0_elliptic_h_recursion](code/general/virasoro/vendor/genus0_elliptic_h_recursion/README.md) | Four-point c=25 elliptic blocks and the sphere multipoint elliptic recursion used in degeneration limits. |
| Matter partition functions | [free_boson_plumbing.py](code/general/scalar/free_boson_plumbing.py), [genus1_compact_boson.py](code/general/scalar/genus1_compact_boson.py) | Free-boson plumbing factors and compact-boson thermal contributions. |
| Normalization and observables | [c1_string_conventions.py](code/general/conventions/c1_string_conventions.py), [c1_observables.py](code/general/conventions/c1_observables.py), [genus2_integrand_normalization.py](code/general/conventions/genus2_integrand_normalization.py) | Convert reduced integrals into amplitudes and free energy with the stated coupling and normalization conventions. |

The coupling convention is `mu^-1 = 2*pi*g_s`. Amplitude outputs have
`g_s^(n-1+2h)` stripped; the genus-two output is the coefficient of `g_s^2` in
`F = -log(Z)/(2*pi*R)`. Tree thermal curves use `R=1/t` for outgoing imaginary
energy t. The torus 1→2 production variable t is the incoming imaginary energy,
so its equal-outgoing-energy plot uses `t/2` and its thermal mapping is `R=2/t`.

## Using the repository

### Recover the draft results from lightweight data

With access to this private GitHub repository, run from a macOS/Linux shell:

```sh
git clone git@github.com:YutaiZhang2003/c1-computation.git
cd c1-computation
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -B reproduce.py
```

Python 3.11 is the tested version. `reproduce.py` verifies source/data hashes,
recovers six result groups, compares 21 reference tables and covariance files,
reconstructs the channel table, and renders eight numerical panels. It writes
`output/recovered/` and ends with `PASS`. Use a new `--output` directory for
another run, or `--no-figures` to recover only numerical outputs.

The bundled inputs are saved momentum-integrated values, moduli weights,
boundary contributions and independent integral estimates. Recovery recomputes
the reductions and uncertainty propagation. It needs no Cannon connection or
external coefficient banks after the dependencies are installed.

To recover only the current cross-channel table and its retained comparisons:

```sh
python -B run.py recover_channel_convergence \
  --output output/channel-convergence \
  --table-output output/channel-comparison
```

The full-precision CSV is written to
`output/channel-comparison/channel_comparison.csv`; accompanying JSON files
record the settings, verification results, momentum nodes and weights.

### Use individual numerical modules

[`run.py`](run.py) supplies the shared import paths and module aliases:

```sh
python -B run.py --list
python -B run.py check_channel_convergence --help
python -B run.py five_point_production scan -- --help
```

The extra `--` in the last command forwards `--help` to the scan's inner
parser. Library modules can instead be imported after calling
`run.bootstrap()` from a script in the repository root.

The [command and file reference](CODE_REFERENCE.md) contains per-calculation
examples, required production inputs, output locations and the purpose of all
220 Python files. New integrations can require prepared geometry, coefficient
banks and substantial compute; the reference identifies those requirements.

### Data and JSON records

| Location | Contents |
| --- | --- |
| [`data/inputs/`](data/inputs/) | Lightweight integrated inputs for the amplitude and free-energy reductions, with retained convergence records. |
| [`data/reference/`](data/reference/) | Independently retained expected tables and covariance arrays used for verification. |
| [`data/channel_comparison/`](data/channel_comparison/) | Target matrices, baseline geometry/scalar evaluations, N12/n12 Liouville receipts and original source records. |
| [`data/channel_convergence/`](data/channel_convergence/) | Current N12/n16 full-precision table, completed fixed-geometry comparisons and source records. |
| [`coefficient_examples/`](data/inputs/computation_examples/coefficient_examples/README.md) | Small coefficient tables with JSON definitions of momenta, weights, expansion variables and truncation conventions. |
| [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) | Hashes of the packaged code/documentation and the original source/draft provenance. |
| [`PACKAGE_MAP.json`](PACKAGE_MAP.json) | Module names and path mappings used by the launcher; scientific file roles are in the sector `CODE_INDEX.json` files linked above. |

A receipt is a saved JSON evaluation record: its inputs, numerical settings,
result and source identifiers. These records distinguish the settings of an
individual evaluation from a program's default options. Complete production
coefficient banks and the manuscript are excluded from this lightweight repo.
