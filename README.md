# c=1 string computations

Organized numerical code and lightweight data for the current
`c1_higher_genus_new_edit` draft. A fresh clone contains everything needed to
recover its **eight numerical figure panels**, the associated numerical tables,
cross-energy/radius covariance matrices, native-q checks, and the retained
genus-two fit. It also reconstructs a fresh five-row theta/glasses comparison
from the order-12 computation at the period matrices printed in the draft.

Recovery uses saved momentum-integrated values, moduli weights, boundary
contributions, and integral replicates. It recomputes the reductions and checks
them against independently retained reference results. It does **not** repeat
the full Liouville integrations.

The inputs occupy about **60 MB**. No coefficient banks, manuscript sources,
compiled manuscript, historical campaigns, or generated figures are committed.
After installing Python dependencies, reproduction needs no network access,
Cannon connection, external data release, or original working directory.

## Reproduce

Use Python **3.11**; the package was tested in a clean Python **3.11.15**
environment. The scripts require at least Python 3.10. From a fresh clone:

```sh
git clone git@github.com:YutaiZhang2003/c1-computation.git
cd c1-computation
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -B reproduce.py
```

The command verifies the source, data and reference checksums, recovers all six
result groups, compares all **21 reference tables and covariance files**,
reconstructs the five channel-comparison rows from saved partition integrals,
and renders eight panels as both PDF and PNG. It writes:

```text
output/recovered/
├── tree_level/5_point/
├── tree_level/6_point/
├── one_loop/two_point/          # includes native_q/ convergence tables
├── one_loop/three_point/
├── one_loop/three_point_thermal/
├── two_loop/                   # includes channel_comparison/
├── figures/                    # eight PDFs, eight PNGs, panels.json
└── verification.json
```

A successful run ends with `PASS`. `verification.json` records the numerical
checks, library versions, remaining uncertainty scopes, and reference
comparisons. Small floating-point differences are allowed; the final comparison
uses `rtol=2e-10`, `atol=1e-28`, in addition to the checks within each sector.
The figures use the same numerical points and analytic comparison curves as the
draft, rendered with Matplotlib; PDF layout is not expected to be byte-identical
to the manuscript's TeX figures.

The output directory must not already exist. For another run or a data-only run:

```sh
python -B reproduce.py --output output/check2
python -B reproduce.py --no-figures --output output/tables-only
python -B reproduce.py --verify-only
```

Do not use Python's `-O` option: it disables assertions used in numerical checks.

## Result coverage

Paths below are relative to the generated output directory.

| Draft result | Retained numerical inputs | Recovered output |
| --- | --- | --- |
| Sphere 1→4 amplitude and thermal curve | 29 integral means; 15 sets of replicate integrals; uncertainty records | `tree_level/5_point/` |
| Direct-series sphere 1→5 amplitude and thermal curve | 15 energies × 8 common scrambles × 256 weighted samples; separate direct-series, refinement and four-point addback terms | `tree_level/6_point/` |
| One-loop 1→1 thermal curve and noncompact check | Native-q integrated replicate receipts, including order/grid variations | `one_loop/two_point/` |
| One-loop 1→2 amplitude | 20 energies, moduli weights and sample values, two later refinements; 22 plotted evaluations; grid RMS and 16-scramble compact covariance | `one_loop/three_point/` |
| One-loop 1→2 thermal curve | 15 radii, 552,960 weighted sampled contributions, compact contributions and collision addbacks | `one_loop/three_point_thermal/` |
| Genus-two free energy | 71 radii, 6,832 bulk samples, separating kernels, nine independent groups and their covariance | `two_loop/` |
| Fresh theta/glasses comparison | Ten order-12 Liouville/scalar evaluations, exact decimal targets, plumbing coordinates and convergence records | `two_loop/channel_comparison/` |

The eight generated figure stems are:

```text
sphere_1to4_re_amplitude     sphere_1to4_thermal_R
sphere_1to5_im_amplitude     sphere_1to5_thermal_R
torus_1to2_re_amplitude      torus_1to1_thermal_R
genus2_free_energy_F2        torus_1to2_thermal_R
```

Useful numerical checks are:

```text
F2(1) = -0.02616489689240621
empirical Monte Carlo SE = 0.00015726732129047136

F2 fit = -pi/2880 * [a*(1 + R^-4) + b*R^-2]
a = 6.993577593353106  +/- 0.021543644957093537
b = 9.99791851458895   +/- 0.12798345856084992
```

The fit uses the original fixed diagonal-weight estimator and propagates the
full cross-radius covariance. Its comparison curve is
`-pi*(7 + 10/R^2 + 7/R^4)/2880`; that prediction is never used to generate
worldsheet numerical points.

### Conventions and uncertainty

The coupling convention is `mu^-1 = 2*pi*g_s`. Amplitudes are coefficients with
`g_s^(n-1+2h)` stripped. The genus-two output is the coefficient of `g_s^2`
in the thermal free energy, `F = -log Z/(2*pi*R)`.

For the tree amplitudes, the saved `t` is each outgoing imaginary energy and
the thermal mapping is `R=1/t`. For the noncompact one-loop 1→2 data, the saved
`t` is the incoming energy: the amplitude plot uses `t/2`, and its thermal
mapping is `R=2/t`. The genus-two plot has a logarithmic radius axis, base two.

Sampling errors and grid RMS retain their original meanings. Grid RMS is a
sensitivity measure, not a confidence interval. The genus-two result is the
resolved contribution; the residual region `0 < |Omega12| < 1e-5` and total
systematic errors remain unbounded. Finite block order, momentum quadrature,
and thermal boundary remainders are not certified by successful data recovery.

### Theta/glasses comparison

The original unrounded targets and computation records for the draft's
five-row table were not found in the retained local data or checked Cannon
archives. The new computation uses the **printed three-decimal period matrices
as exact targets**, with the current kernels, finite-c recursion order **12**,
and **12 generalized Gauss–Laguerre nodes per momentum integral**. The draft
specifies these block and momentum orders in its genus-two numerical stability
discussion.

Momentum nodes follow the current `primary-gaussian` implementation:
`u_j` are the roots for generalized Laguerre weight `u^-1/2 exp(-u)`, and
each tube uses `P_j = sqrt(u_j / (-2 log |q_edge|))`. Thus the momentum
locations adapt to that tube's plumbing coordinate. The recovered
`channel_momentum_quadrature.json` lists all nodes and integration weights.

This is a fresh comparison. Its absolute Q_L values differ from the old table;
the package does not claim to recover those old entries or attribute their
discrepancy to rounding. The current channel ratios and period-map diagnostics
are calculated in `two_loop/channel_comparison/channel_comparison.csv` and
`channel_comparison_verification.json`. `comparison_with_printed_table.csv`
records the changes in absolute values using a separate transcription of the
printed entries; those entries never enter the partition-function calculation.
The 2026-10-20 rerun gives
`max |Q_L_theta / Q_L_glasses - 1| = 1.2416270324555612e-5`,
within the draft's `2e-5` agreement criterion. The maximum inverse-period
residual is `1.9964218564878783e-10`. The new absolute Q_L values are
54.5–77.0% higher than the printed entries; the origin of that discrepancy
remains unresolved.

[`data/channel_comparison/`](data/channel_comparison/) contains the exact
targets, ten partition-integral receipts, and source manifests and drivers
for the Cannon runs. The recovery recomputes
`log Q_L = log Z_L - 25 log Z_X`, exponentiates it, and forms the channel
ratios. It verifies that both channels use the same target, checks the integer
symplectic marking, and checks the saved period residuals against `2e-10`.
The scalar follows the dimensionless handle-momentum convention implemented
in `free_boson_plumbing.py`.

The full comparison can be rerun without coefficient banks or external data.
Each command computes one row in one channel and may take substantial CPU
time. Run all ten combinations of rows 1–5 and channels `theta`, `glasses`:

```sh
python -B run.py recompute_channel_comparison evaluate \
  --row 1 --channel theta --block-order 12 --nodes 12 \
  --output output/channel-rerun
# Repeat for the other nine row/channel pairs, then:
python -B run.py recompute_channel_comparison assemble \
  --receipts output/channel-rerun --output output/channel-rerun-table
```

The evaluator resums the global block, uses 50-digit DOZZ arithmetic,
and saves the scalar and vacuum-product settings. The glasses scalar is
checked at successive word caps 8, 10, …, 16, requiring two successive changes
in `Z_X^-25` below `1e-8`; this is an empirical convergence check. The scalar
mode cap is 120, and the vacuum seed has word cap 12 and oscillator cap 40.
The Slurm template
[`channel_comparison.sbatch`](code/two_loop/other_scripts/channel_comparison.sbatch)
launches the ten evaluations with an explicitly supplied run directory and
Python executable.

Other numerical-appendix receipts are retained under
[`data/inputs/convergence/`](data/inputs/convergence/). The one-loop 1→1 order
comparisons are recomputed by the main command; the other supplementary
receipts are checksum-verified evidence, rather than fresh convergence runs.

## Code organization

| Location | Purpose |
| --- | --- |
| [`code/tree_level/five_point/`](code/tree_level/five_point/) | Sphere 1→4 blocks, Liouville quadrature, moduli integration, production and checks |
| [`code/tree_level/six_point/`](code/tree_level/six_point/) | Direct six-point series, separating-region treatment, sampling and refinements |
| [`code/one_loop/two_point/`](code/one_loop/two_point/) | Native-q thermal/noncompact 1→1 calculations |
| [`code/one_loop/three_point/`](code/one_loop/three_point/) | Noncompact and thermal 1→2 calculations |
| [`code/two_loop/`](code/two_loop/) | Period maps, Liouville/scalar/ghost factors and genus-two integration |
| [`code/general/`](code/general/) | Shared conventions, Virasoro and Liouville routines, geometry and I/O |
| [`code/general/recovery/`](code/general/recovery/) | Lightweight data construction and result recovery |
| [`code/figures/plot_recovered_results.py`](code/figures/plot_recovered_results.py) | Eight numerical panels from recovered tables |
| [`data/inputs/`](data/inputs/) | Inputs to the numerical reductions |
| [`data/reference/`](data/reference/) | Independent expected results used only for verification |
| [`data/channel_comparison/`](data/channel_comparison/) | Fresh order-12 channel-comparison inputs and receipts |

Each calculation separates conformal blocks, momentum integration, moduli
integration and production/support scripts. `CODE_INDEX.json` and
`CODE_INDEX.tsv` describe the retained source files; `run.py` resolves the
shared module paths and compatibility aliases:

```sh
python -B run.py --list
python -B run.py five_point_production scan --help
python -B run.py six_point_refinement continue --help
python -B run.py genus1_two_point_thermal_production --help
python -B run.py genus1_final_curve_plan --help
python -B run.py genus2_moduli_production --help
```

These are computational implementations, not a promise of an immediate full
production rerun. New integrations require explicitly prepared geometry,
coefficient banks, run settings and substantial compute. Some production
launchers retain original cluster defaults. The supported standalone recovery
entry point is `reproduce.py`; older campaign exporters may require campaign
files that are intentionally absent from this repository.

## Lightweight data and provenance

The original retained input dataset was approximately 227 MB. Compaction:

* Stores the one-loop three-point quadrature rules as ordered numeric index and
  weight arrays. Sample values, double precision, duplicate entries and
  summation order are preserved.
* Removes unused thermal unit-cube coordinates and derived coordinates while
  retaining every weighted value, arithmetic bound, scramble and seed.
* Omits verbose genus-two geometry CSV and collision-mesh descriptions whose
  numerical values and weights are already retained.
* Keeps convergence receipts and small coefficient examples.

[`data/inputs/DATASET_MANIFEST.json`](data/inputs/DATASET_MANIFEST.json) records
every input hash, original hashes, transformations and omitted files.
[`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) identifies the source snapshot,
draft SHA-256 and current packaged code.
[`data/reference/RESULTS_MANIFEST.json`](data/reference/RESULTS_MANIFEST.json)
pins expected results. Reference source-path labels were made relative;
numerical fields were preserved.

For maintainers with the original retained dataset, the compaction can be
repeated into a new directory:

```sh
python -B code/general/recovery/build_light_dataset.py \
  --source /path/to/production_integrated \
  --output /tmp/c1-light-inputs
```

This maintainer command is unnecessary for users of a fresh clone.
