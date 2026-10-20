# c=1 string computations

This repository contains the numerical code and lightweight data supporting the
`c1_higher_genus_new_edit` draft: sphere 1→4 and 1→5 amplitudes, torus 1→1 and
1→2 amplitudes, their thermal curves, and the genus-two free energy.

**Start with `python -B reproduce.py`.** It recovers the draft's eight numerical
figure panels, numerical tables, covariance matrices, native-q checks and
genus-two fit from the bundled inputs. It also reconstructs the current five-row
theta/glasses table at block order 12 with 16 momentum nodes per edge.

Recovery uses saved momentum-integrated values, moduli weights, boundary
contributions, and integral replicates. It recomputes the reductions and checks
them against independently retained reference results. It does **not** repeat
the full Liouville integrations.

The inputs occupy about **60 MB**. No coefficient banks, manuscript sources,
compiled manuscript, historical campaigns, or generated figures are committed.
After installing Python dependencies, reproduction needs no network access,
Cannon connection, external data release, or original working directory.

The [code catalogue](#file-by-file-code-catalogue) describes every Python file
under `code/`. The [usage guide](#using-the-numerical-codes) explains the common
launcher and library interface; [new-calculation examples](#starting-new-calculations)
identify the inputs required by production programs.

**Contents:** [Setup](#setup-and-quick-start) · [Choose a program](#which-program-should-i-use)
· [Results](#result-coverage) · [Channel matching](#thetaglasses-comparison)
· [Code usage](#using-the-numerical-codes) · [New calculations](#starting-new-calculations)
· [Provenance](#lightweight-data-and-provenance) · [File catalogue](#file-by-file-code-catalogue)

## Setup and quick start

Use Python **3.11**; the package was tested in a clean Python **3.11.15**
environment. The scripts require at least Python 3.10. The commands below use
a macOS/Linux shell and run from the repository root. This is a private
repository, so cloning requires an authorized GitHub account and SSH access.

```sh
git clone git@github.com:YutaiZhang2003/c1-computation.git
cd c1-computation
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -B reproduce.py
```

`requirements.txt` pins the tested NumPy, SciPy, mpmath, SymPy, Matplotlib and
Pillow versions. No separate installation of the shared numerical modules is
needed for these commands.

The reproduction command verifies the source, data and reference checksums, recovers all six
result groups, compares all **21 reference tables and covariance files**,
reconstructs the five channel-comparison rows from saved partition integrals,
recovers their fixed-geometry convergence comparisons,
and renders eight panels as both PDF and PNG. It writes:

```text
output/recovered/
├── tree_level/5_point/
├── tree_level/6_point/
├── one_loop/two_point/          # includes native_q/ convergence tables
├── one_loop/three_point/
├── one_loop/three_point_thermal/
├── two_loop/
│   ├── channel_comparison/     # current N=12, 16-node table
│   ├── channel_comparison_baseline_N12_n12/
│   └── channel_convergence/    # completed checks and explicit missing checks
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

`--verify-only` checks file integrity; it does not recalculate or compare
numerical results. `--no-figures` performs the numerical recovery and comparisons
while skipping rendering. To use the environment in a new terminal, first run
`. .venv/bin/activate` from this directory.

Do not use Python's `-O` option: it disables assertions used in numerical checks.

## Which program should I use?

| Program | Purpose | Inputs → outputs | How to use it |
| --- | --- | --- | --- |
| [`reproduce.py`](reproduce.py) | Complete, verified recovery of the draft results | Bundled data and manifests → tables, covariance arrays, eight figures and `verification.json` | `python -B reproduce.py` |
| [`run.py`](run.py) | Find modules, set up imports and launch an individual program | Module name and its arguments → that program's outputs | `python -B run.py --list`; see [launcher usage](#using-the-numerical-codes) |
| [`recover_production_results.py`](code/general/recovery/recover_production_results.py) | Recover the six amplitude/free-energy groups from saved integrated data | `data/inputs/` → sector tables and optional figures | Called by `reproduce.py`; its lower-level CLI accepts `--data-dir`, `--output` and `--figures` |
| [`recover_native_q_results.py`](code/general/recovery/recover_native_q_results.py) | Recover torus 1→1 native-q scans and retained order comparisons | Saved native-q receipts → comparison tables | Library used by the main recovery |
| [`recover_channel_convergence.py`](code/two_loop/other_scripts/recover_channel_convergence.py) | Recover the current channel table and completed convergence checks | Saved N12/n16 and N18/n12 receipts plus the baseline → full-precision CSV and verification JSON | Use the [channel recovery command](#thetaglasses-comparison) |
| [`check_channel_convergence.py`](code/two_loop/other_scripts/check_channel_convergence.py) | Recompute one Liouville integral at a saved geometry and scalar normalization | Row, channel, block order and momentum order → one new JSON receipt | Use `--dry-run` to inspect a job; see [fresh channel evaluations](#fresh-channel-evaluations) |
| [`recompute_channel_comparison.py`](code/two_loop/other_scripts/recompute_channel_comparison.py) | Solve the target geometry, evaluate scalar and Liouville factors, or assemble the N12/n12 baseline | Target matrices → per-channel receipts; baseline receipts → table | `evaluate` and `assemble` subcommands; see [fresh channel evaluations](#fresh-channel-evaluations) |
| [`plot_recovered_results.py`](code/figures/plot_recovered_results.py) | Draw the eight panels from recovered numerical tables | Recovered output directory → `figures/*.pdf`, `*.png`, `panels.json` | Library called by `reproduce.py`; see [redrawing figures](#redrawing-figures) |
| [`build_light_dataset.py`](code/general/recovery/build_light_dataset.py) | Construct the compact dataset from the original production archive | Original `production_integrated/` → a new lightweight input directory | Maintainer command in [provenance](#lightweight-data-and-provenance) |
| [`refresh_source_manifest.py`](code/general/recovery/refresh_source_manifest.py) | Record an intentional source/documentation update and refresh source-index hashes | Current repository files → `SOURCE_MANIFEST.json` and source indexes | Maintainer command in [provenance](#lightweight-data-and-provenance) |

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
| Fresh theta/glasses comparison | Ten order-12, 16-node Liouville evaluations, scalar normalization, exact decimal targets and convergence records | `two_loop/channel_comparison/` |

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

The current draft table uses finite-c recursion order **12** and **16
primary-Gaussian momentum nodes per edge**, at the five period matrices printed
in the draft. The exact decimal targets are in
[`targets.json`](data/channel_comparison/targets.json). The original unrounded
coordinates were unavailable, so these printed decimals define the targets.
Both channels use the same targets, with their saved symplectic markings and
certified inverse-period maps.

The full-precision table is
[`current_channel_comparison.csv`](data/channel_convergence/current_channel_comparison.csv).
It contains both Q_L values, the theta/glasses ratio, and the signed relative
difference for every row. The maximum absolute relative difference is
`6.958719225201529e-8`, supporting cross-channel agreement within `1e-7` at these
five matrices. This is empirical channel agreement, not an absolute error bound.
The CSV and JSON receipts retain the full stored floating-point precision;
rounded manuscript transcriptions are not included.

The ten `row*_N12_n16_dps50.json` receipts under
[`data/channel_convergence/receipts/`](data/channel_convergence/receipts/)
contain the Liouville integrals, scalar normalization, plumbing coordinates,
period-map diagnostics, numerical settings and source hashes. Recovery computes
`log Q_L = log Z_L - 25 log Z_X`, exponentiates it, and forms ratios from the
unrounded values. The scalar follows the dimensionless handle-momentum
convention in `free_boson_plumbing.py`. The `dps=50` setting applies to DOZZ
arithmetic; stored partition values use binary64 precision.

The default `python -B reproduce.py` exports the current 16-node table to
`two_loop/channel_comparison/channel_comparison.csv`, together with verification
metadata and all 30 momentum-node and weight lists. The earlier full-precision
12-node receipts remain under [`data/channel_comparison/`](data/channel_comparison/)
as the baseline for convergence comparisons; their recovered table is written
to `two_loop/channel_comparison_baseline_N12_n12/`.

To recover only the current comparison and its retained convergence evidence,
without running new Liouville integrations:

```sh
python -B run.py recover_channel_convergence \
  --output output/channel-convergence \
  --table-output output/channel-comparison
```

The two output directories contain:

| File | Meaning |
| --- | --- |
| `output/channel-comparison/channel_comparison.csv` | Five rows: `Q_L_glasses`, `Q_L_theta`, `theta_over_glasses`, and the signed `relative_difference = Q_L_theta / Q_L_glasses - 1` |
| `output/channel-comparison/channel_comparison_verification.json` | Settings, maximum channel difference and precision scope |
| `output/channel-comparison/channel_momentum_quadrature.json` | Momentum nodes and weights for all three edges in both channels at every target |
| `output/channel-convergence/convergence.csv` | Completed quadrature/block-order comparisons with their baseline |
| `output/channel-convergence/convergence.json` | Full recovery report, source checks and the list of missing evaluations |

Ratios are formed from logarithms of unrounded integrals, and the signed
difference uses `expm1(log_Q_L_theta - log_Q_L_glasses)` to avoid subtracting
nearly equal rounded values. The stored CSV is the reference export; recovery
reconstructs its numbers from the JSON receipts and verifies agreement.

Momentum nodes follow the current `primary-gaussian` implementation: `u_j` are
the generalized Laguerre roots for weight `u^-1/2 exp(-u)`, and each edge uses
`P_j = sqrt(u_j / (-2 log |q_edge|))`. This covers the infinite momentum domain
without a finite cutoff. Numerical kernels are verified against the baseline
source snapshots. The evaluator resums the global block, retains
the vacuum seed at word cap 12 and oscillator cap 40, and uses 50-digit DOZZ
arithmetic. The saved scalar evaluations use mode cap 120. The glasses scalar
was checked at word caps 8, 10, …, 16, requiring two successive changes in
`Z_X^-25` below `1e-8`; the period residuals are below `2e-10` in each channel's
homology basis. The source and baseline scalar/geometry evaluations can also be
rerun with `recompute_channel_comparison evaluate`; see its `--help` output.

The 2026-10-20 convergence campaign was stopped at the author's request after
all ten 16-node evaluations, all five order-18 glasses evaluations, two
order-18 theta evaluations (rows 1 and 3), and the baseline control finished.
The other three theta order-18 jobs were cancelled; missing checks remain
explicit in the recovered report. Increasing momentum nodes from 12 to 16
changes theta by at most `1.2625239402283483e-5` and glasses by at most
`2.868510137674608e-7`. The completed order-12 to order-18 comparisons change
glasses by at most `3.828885224405717e-8`; the two completed theta comparisons
show no change at the stored precision. These checks hold geometry and scalar
normalization fixed and do not give a rigorous truncation-error bound.

Other numerical-appendix receipts are retained under
[`data/inputs/convergence/`](data/inputs/convergence/). The one-loop 1→1 order
comparisons are recomputed by the main command; the other supplementary
receipts are checksum-verified evidence, rather than fresh convergence runs.

## Using the numerical codes

The calculation flows through four stages: construct conformal-block
coefficients; integrate the Liouville momenta with DOZZ factors; integrate
over moduli with the required boundary terms; assemble observables and their
uncertainties. The recovery workflow starts from the saved momentum integrals
and recomputes the remaining reductions. Plotting uses those recovered results.

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
| [`data/channel_comparison/`](data/channel_comparison/) | Exact decimal targets, N12/n12 baseline integrals, scalar factors and geometry |
| [`data/channel_convergence/`](data/channel_convergence/) | Current N12/n16 table, integral receipts and completed block-order checks |

### Launching a program

Use `run.py` from the repository root for programs under `code/`. It resolves
shared imports and historical module aliases. Pass the filename without `.py`,
followed by the program's arguments. For programs with subcommands, put the
subcommand before its options:

```sh
# List available module names and their source paths.
python -B run.py --list

# Inspect a program's options without launching a calculation.
python -B run.py check_channel_convergence --help

# Inspect the scan's options; -- forwards --help past its outer parser.
python -B run.py five_point_production scan -- --help
```

The `five_point_production` and `six_point_refinement` controllers need the
extra `--` before `--help` to display the inner command's options. Without it,
they display only the outer command selector.

The catalogue below distinguishes **CLI** programs, importable **Library**
modules, and **Script** entry points that run without an argument parser.
`--help` is for CLI programs; it will not stop a Script from executing.

Write new results under `output/` or to an external directory. `run.py` protects
the packaged source and data from writes. If a production program writes to
default data paths, use its explicit output option or an isolated workspace:

```sh
# Creates a sibling workspace and copies the bundled data there once.
# This invocation only displays the scan options.
python -B run.py --work-dir ../c1-workspace five_point_production scan -- --help
```

`--work-dir` is a launcher option and must appear **before** the module name.
Its directory must be outside this repository and its ancestors. The copied
lightweight dataset does not supply omitted production coefficient banks or
historical campaign directories. Check a production command's explicit input
paths before running it.

### Using a library in Python

Library modules expose functions for other calculations. Start a Python script
from the repository root and call `run.bootstrap()` before importing them.
For example, this constructs a 16-node momentum rule for an illustrative edge
with `q=0.01`; it does not perform a Liouville integral:

```sh
python -B - <<'PY'
import run
run.bootstrap()
from liouville_momentum_quadrature import primary_gaussian_momentum_rule

rule = primary_gaussian_momentum_rule(0.01 + 0j, 16)
for momentum, weight in zip(rule.nodes, rule.weights):
    print(f"{momentum!r},{weight!r}")
PY
```

Function signatures and docstrings in the linked files specify their
arguments, normalization and return types. The vendored
`genus0_elliptic_h_recursion` package has its own
[API guide](code/general/virasoro/vendor/genus0_elliptic_h_recursion/README.md).

### Redrawing figures

After a successful recovery, redraw all eight panels from its tables:

```sh
python -B - <<'PY'
from pathlib import Path
import run
run.bootstrap()
from plot_recovered_results import render_all

render_all(Path("output/recovered"))
PY
```

Change the path if recovery used a different `--output` directory. This replaces
the figure files inside that directory and leaves the numerical tables intact.

## Starting new calculations

These workflows run new numerical work and can require substantial CPU time.
The commands in the quick start recover existing data. The following examples
are separate, optional research workflows; their default or illustrative
settings should not be treated as a complete specification of the draft's
production campaigns.

### Fresh channel evaluations

Inspect the saved geometry, settings and source consistency without evaluating
an integral:

```sh
python -B run.py check_channel_convergence \
  --row 1 --channel theta --block-order 12 --nodes 16 --dps 50 \
  --output output/channel-rerun --dry-run
```

To run that one integral, remove `--dry-run`. Its output is
`output/channel-rerun/row1_theta_N12_n16_dps50.json`. Choose `--row` from 1–5 and
`--channel` from `theta` or `glasses`. The saved period matrix, plumbing
coordinates, homology marking and scalar factor stay fixed. `--block-order`
sets the finite-c recursion order, `--nodes` sets the momentum nodes per edge,
and `--dps` sets the DOZZ arithmetic precision. An existing receipt with the
same filename is rejected.

For a calculation that also solves the inverse period map and recomputes the
scalar factor, inspect the full evaluator:

```sh
python -B run.py recompute_channel_comparison evaluate --help
```

The following recomputes one baseline entry, including geometry and scalar:

```sh
python -B run.py recompute_channel_comparison evaluate \
  --row 1 --channel theta --block-order 12 --nodes 12 --dps 50 \
  --output output/channel-baseline-rerun
```

Its optional `--targets` accepts the same JSON schema as
[`targets.json`](data/channel_comparison/targets.json).
After all ten **N12/n12** baseline evaluations are complete, assemble them with:

```sh
python -B run.py recompute_channel_comparison assemble \
  --receipts output/channel-baseline-rerun \
  --output output/channel-baseline-table
```

The `assemble` CLI expects the baseline receipt format and N12/n12 filenames.
Use `recover_channel_convergence --table-output ...` for the packaged current
N12/n16 results. Its recovery validates the saved campaign and its source
manifests; it is not a generic collector for an arbitrary new campaign.

### Sphere amplitudes

For **sphere 1→4**, the four core modules implement the channel atlas,
Virasoro recursion, momentum integrand and moduli estimator.
`five_point_production scan` combines them into an energy scan, including
contour residues when requested. This is an example of a new scan at one
energy, with explicit output and numerical settings:

```sh
python -B run.py five_point_production scan \
  --t 0.19 --block-scheme c --block-order 6 \
  --momentum-order 16 --momentum-maximum 6 \
  --liouville-contour continued --sobol-power 10 --replicates 4 \
  --output output/sphere-five-scan.json
```

`--sobol-power 10` means 2^10 moduli samples per replicate. The JSON records
the integral, sampling errors and amplitude convention. Further energy points,
block orders and contour choices are controlled by the CLI; use the bundled
recovery for the exact plotted dataset.

For **sphere 1→5**, the production chain requires a prepared geometry release
and verified momentum coefficient banks. Its stages are:

| Stage | Programs | Required input → output |
| --- | --- | --- |
| Prepare geometry | `six_point_prepare_radial_geometry` | Sampling settings and optional reuse records → geometry shards and manifest |
| Prepare momentum data | `six_point_mixed_order_node_catalog`, `six_point_build_momentum_banks` | Explicit momentum rules and block orders → node catalogue and coefficient banks |
| Prepare and evaluate curve | `six_point_prepare_amplitude_curve`, `six_point_run_frozen_radial_worker` | Accepted production release, geometry and banks → per-energy integration records |
| Refine selected blocks | `six_point_refinement` (`initial`, `continue`) | Existing curve and selected nodes → higher-order corrections |
| Assemble results | `six_point_refinement_assembly`, `six_point_assemble_radial_curve_covariance` | Completed baseline and refinement records → amplitudes and shared-scramble covariance |

Inspect the preparation and continuation interfaces with:

```sh
python -B run.py six_point_prepare_amplitude_curve --help
python -B run.py six_point_refinement continue -- --help
```

The accepted production release and complete coefficient banks are not in this
lightweight repository. The small
[coefficient examples](data/inputs/computation_examples/coefficient_examples/README.md)
document the conventions and are insufficient to launch the full campaign.

### Torus amplitudes

For **torus 1→1**, `genus1_two_point_thermal_production` builds the required
Liouville coefficient banks, runs the thermal moduli integrals and saves a
receipt for each radius. The controller uses the retained
[`production_settings.json`](code/one_loop/two_point/production_settings.json),
with CLI overrides. For a new calculation at R=2:

```sh
python -B run.py genus1_two_point_thermal_production \
  --radii 2 --workers 1 --output-dir output/torus-two-rerun
```

The current real-contour implementation requires `R > 1`. Coefficient banks
go to `output/torus-two-rerun/liouville/`; the integral receipt and log go to
`output/torus-two-rerun/scan_native_q_complete_collision/`.
`genus1_two_point_thermal_report` assembles those receipts into amplitude tables.
For the single-radius calculation above:

```sh
python -B run.py genus1_two_point_thermal_report \
  --scan-dir output/torus-two-rerun/scan_native_q_complete_collision \
  --radii 2 --output-dir output/torus-two-report
```

Add `--plot` when assembling a radius scan to render its radius figure.

For **torus 1→2**, `genus1_moduli_amplitude` provides `prepare`, `worker` and
`reduce` stages using a pinned set of CFT banks, geometry and boundary terms.
`genus1_final_curve_plan` prepares the twenty-energy design from a previous
curve and pair-cell refinement. `draft_radius_campaign` computes paired
thermal corrections; `draft_radius_collect` combines them with the baseline.

```sh
python -B run.py genus1_moduli_amplitude --help
python -B run.py genus1_final_curve_plan --help
python -B run.py draft_radius_campaign --help
```

These controllers require the original prepared banks and campaign inputs.
Several thermal and plotting scripts retain historical directory defaults.
The standalone recovery instead uses the bundled integrated values to recover
both torus 1→2 curves and their uncertainty/covariance outputs.

### Genus-two moduli integration

`genus2_moduli_production` reads a prepared CSV of moduli nodes, selects and
validates a plumbing chart, evaluates Liouville/scalar/ghost factors and writes
per-node estimates. `genus2_separating_production` handles the separating
region and its controls. The `assemble_genus2_*` programs combine completed
bulk and separating contributions; `plot_genus2_absolute_bry` performs the
radius fit and plotting for its original campaign format.

```sh
python -B run.py genus2_moduli_production --help
python -B run.py genus2_separating_production --help
```

New production requires explicit geometry/input paths and numerical settings.
The original verbose geometry release is omitted from the compact package;
the retained node values, weights and independent groups are sufficient for
`reproduce.py` to recover the reported free-energy curve and covariance.

### Cluster templates

| Template | Purpose and required setup |
| --- | --- |
| [`channel_comparison.sbatch`](code/two_loop/other_scripts/channel_comparison.sbatch) | Ten N12/n12 evaluations, solving geometry and scalar factors. Set `C1_COMPARISON_ROOT` to a new run directory containing this repository at `source/`, and `C1_PYTHON` to an interpreter with the dependencies. Receipts go to `C1_COMPARISON_ROOT/receipts/`. |
| [`channel_convergence.sbatch`](code/two_loop/other_scripts/channel_convergence.sbatch) | Twenty-one fixed-geometry tasks: ten N12/n16, ten N18/n12 and one N12/n12 control. Set `C1_CONVERGENCE_ROOT` with a `source/` checkout and `C1_PYTHON`; receipts go to `C1_CONVERGENCE_ROOT/receipts/`. |

Both are Slurm templates with the original Cannon partition/resource defaults.
Review those settings for the target cluster. The completed campaign has
18 retained receipts; the three cancelled checks are not required for recovery.

## Common usage issues

| Message or situation | Action |
| --- | --- |
| Python cannot import a shared numerical module | Activate `.venv`, run from the repository root, and use `run.py` or `run.bootstrap()` as shown above. |
| `The output directory already exists` | Give `reproduce.py` a new `--output` directory. |
| `Source and data are immutable` | Set an output under `output/`, or use an external `--work-dir` before the module name. |
| `Changed package file` or a checksum mismatch | Compare the changed file with the committed version. Refresh a manifest only for an intentional reviewed source edit; do not suppress an unexplained data mismatch. |
| A production script asks for a missing bank/campaign file | Supply the prepared production input, or use `reproduce.py` to recover the supported results from bundled integrated data. |
| `--verify-only` succeeds but there are no plots | Run `python -B reproduce.py` to recover and render; integrity verification alone produces no figures. |

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

After an intentional source or documentation edit, maintainers update the
source inventory and then check it:

```sh
python -B code/general/recovery/refresh_source_manifest.py
python -B reproduce.py --verify-only
```

Run the manifest updater directly as shown, because it intentionally writes
inside the source snapshot. It does not update the data or reference manifests.

## File-by-file code catalogue

The expandable tables describe all **220 Python files under `code/`**. The
root entry points `reproduce.py` and `run.py` are described in
[Which program should I use?](#which-program-should-i-use); the two Slurm scripts
are described under [Cluster templates](#cluster-templates).

**Interface key:** **CLI** means `python -B run.py NAME --help`, with `NAME`
equal to the filename without `.py`. **Library** means import after
`run.bootstrap()` and call its documented functions. **Script** means an
entry point without a help parser: inspect its source/configuration before
running `python -B run.py NAME`. **Maintainer** uses the direct command above;
**Package demo** follows the vendored package's own README. A CLI label describes
the interface; required production banks or campaign files may still be absent.

Within each calculation, `conformal_block/` contains block construction,
`liouville_integration/` momentum integrals, `moduli_integration/` geometry,
weights and boundary terms, and `production/` or `other_scripts/` controllers,
assembly and checks. Linked filenames open the implementation. `CODE_INDEX.json`
and `CODE_INDEX.tsv` in the sector directories retain the source hashes.

<details>
<summary>Shared numerical tools and import support — 34 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [conventions/c1_matrix_model.py](code/general/conventions/c1_matrix_model.py) | Evaluate matrix-model comparison curves after worldsheet integration. | Library |
| [conventions/c1_observables.py](code/general/conventions/c1_observables.py) | Convert reduced integrals to final amplitude and free-energy observables. | Library |
| [conventions/c1_string_conventions.py](code/general/conventions/c1_string_conventions.py) | Define BRY amplitude normalization and coupling conversions. | Library |
| [conventions/genus2_integrand_normalization.py](code/general/conventions/genus2_integrand_normalization.py) | Apply and audit the independent normalization factors in the genus-two vacuum integrand. | Library |
| [geometry/ccy_plumbing_conventions.py](code/general/geometry/ccy_plumbing_conventions.py) | Cho-Collier-Yin plumbing-frame sewing conventions. | Library |
| [geometry/conformal_frame_labels.py](code/general/geometry/conformal_frame_labels.py) | Shared conformal-frame labels for the genus-two partition functions. | Library |
| [geometry/genus3_plumbing_channels.py](code/general/geometry/genus3_plumbing_channels.py) | The five marked trivalent genus-three plumbing channels. | Library |
| [geometry/plumbing_algorithms.py](code/general/geometry/plumbing_algorithms.py) | Core plumbing algorithms for holomorphic one-forms. | Library |
| [io/c1_frozen_bundle.py](code/general/io/c1_frozen_bundle.py) | Verify the required production files and hashes of an original frozen bundle. | Library |
| [io/c1_paths.py](code/general/io/c1_paths.py) | Resolve package modules, retained data paths and optional writable workspaces. | Library |
| [liouville/liouville_torus.py](code/general/liouville/liouville_torus.py) | Liouville torus one-point functions from Virasoro blocks. | CLI |
| [numerics/liouville_momentum_quadrature.py](code/general/numerics/liouville_momentum_quadrature.py) | Momentum quadrature helpers for Liouville plumbing integrals. | Library |
| [numerics/torus_sampling.py](code/general/numerics/torus_sampling.py) | Torus proposal and lattice terms used by the genus-two primary control. | Library |
| [recovery/build_light_dataset.py](code/general/recovery/build_light_dataset.py) | Build the lightweight inputs from an existing production_integrated directory. | CLI |
| [recovery/recover_native_q_results.py](code/general/recovery/recover_native_q_results.py) | Recover current native-q scans and the order checks cited in the draft. | Library |
| [recovery/recover_production_results.py](code/general/recovery/recover_production_results.py) | Recover the draft's numerical curves from the bundled lightweight inputs. | CLI |
| [recovery/refresh_source_manifest.py](code/general/recovery/refresh_source_manifest.py) | Refresh the source manifest after intentional source edits (maintainers only). | Maintainer |
| [scalar/free_boson_pair_of_pants.py](code/general/scalar/free_boson_pair_of_pants.py) | Direct rank-one Heisenberg-VOA sewing on trivalent plumbing graphs. | CLI |
| [scalar/free_boson_plumbing.py](code/general/scalar/free_boson_plumbing.py) | Free-boson plumbing diagnostics in genus two and genus three. | CLI |
| [scalar/genus1_compact_boson.py](code/general/scalar/genus1_compact_boson.py) | Compact Euclidean boson factor relative to the noncompact torus integrand. | Script |
| [virasoro/ccy_genus2_block.py](code/general/virasoro/ccy_genus2_block.py) | Cho-Collier-Yin genus-two Virasoro block in the plumbing frame. | CLI |
| [virasoro/genus2_vacuum_blocks.py](code/general/virasoro/genus2_vacuum_blocks.py) | Genus-two vacuum Virasoro block tools. | CLI |
| [virasoro/sphere_four_point_c25.py](code/general/virasoro/sphere_four_point_c25.py) | Analytically grouped four-point elliptic h recursion at exactly c=25. | Library |
| [virasoro/sphere_four_point_pillow_evaluate.py](code/general/virasoro/sphere_four_point_pillow_evaluate.py) | Analytic z derivatives of a fixed four-point elliptic block polynomial. | Library |
| [virasoro/torus_descendant_blocks.py](code/general/virasoro/torus_descendant_blocks.py) | Torus one-point blocks with Virasoro-descendant external insertions. | CLI |
| [virasoro/torus_primary_low_order.py](code/general/virasoro/torus_primary_low_order.py) | Vectorized c=25 torus primary elliptic H through level four. | Library |
| [virasoro/virasoro_blocks.py](code/general/virasoro/virasoro_blocks.py) | Virasoro torus one-point and multipoint necklace conformal blocks. | CLI |
| [virasoro/virasoro_descendant_algebra.py](code/general/virasoro/virasoro_descendant_algebra.py) | Low-level Virasoro descendant algebra in a PBW basis. | Library |
| [virasoro/virasoro_plumbing_graph.py](code/general/virasoro/virasoro_plumbing_graph.py) | Direct Virasoro conformal blocks for trivalent plumbing graphs. | Library |
| [one_loop/genus1/__init__.py](code/one_loop/genus1/__init__.py) | Compatibility namespace for genus-one production imports. | Library |
| [one_loop/genus1/project.py](code/one_loop/genus1/project.py) | Install the classified genus-one source directories for compatibility imports. | Library |
| [tree_level/general/momentum_quadrature.py](code/tree_level/general/momentum_quadrature.py) | Finite-interval Liouville momentum quadrature shared by tree amplitudes. | Library |
| [tree_level/general/sphere_mobius_geometry.py](code/tree_level/general/sphere_mobius_geometry.py) | Projective-sphere coordinates and Möbius maps, independent of multiplicity. | Library |
| [tree_level/general/tree_level_source_compatibility.py](code/tree_level/general/tree_level_source_compatibility.py) | Resolve historical module names and verify recorded source-only reorganizations. | Library |

</details>

<details>
<summary>Sphere 1→4 — 14 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [conformal_block/five_point_liouville_correlator_and_atlas.py](code/tree_level/five_point/conformal_block/five_point_liouville_correlator_and_atlas.py) | Five-point c=25 Liouville correlator in the CCY linear-channel atlas. | Library |
| [conformal_block/five_point_virasoro_block_recursion.py](code/tree_level/five_point/conformal_block/five_point_virasoro_block_recursion.py) | CCY recursions for the sphere five-point Virasoro block. | Library |
| [liouville_integration/five_point_integrand_and_quadrature.py](code/tree_level/five_point/liouville_integration/five_point_integrand_and_quadrature.py) | Reusable equal-energy kernel for the sphere five-point worldsheet integral. | Library |
| [moduli_integration/five_point_moduli_integration.py](code/tree_level/five_point/moduli_integration/five_point_moduli_integration.py) | Sampling densities, weighted moduli integration, and independent-scramble errors. | Library |
| [other_scripts/ccy_sphere_five_point_checks.py](code/tree_level/five_point/other_scripts/ccy_sphere_five_point_checks.py) | Independent checks of the CCY sphere five-point linear-channel block. | Script |
| [other_scripts/export_sphere_five_point_t_axis_plot_data.py](code/tree_level/five_point/other_scripts/export_sphere_five_point_t_axis_plot_data.py) | Export the frozen sphere 1->4 points for the t-axis amplitude figure. | CLI |
| [other_scripts/five_point_production.py](code/tree_level/five_point/other_scripts/five_point_production.py) | Sphere 1->4 scan and the frozen twelve-point extension. | CLI |
| [other_scripts/sphere_five_point_30point_audit_summary.py](code/tree_level/five_point/other_scripts/sphere_five_point_30point_audit_summary.py) | Freeze target-free block-order audits of representative new 1->4 points. | CLI |
| [other_scripts/sphere_five_point_30point_checks.py](code/tree_level/five_point/other_scripts/sphere_five_point_30point_checks.py) | Regression checks for the sphere `1->4` 30-point campaign artifacts. | Script |
| [other_scripts/sphere_five_point_30point_matrix_comparison.py](code/tree_level/five_point/other_scripts/sphere_five_point_30point_matrix_comparison.py) | Compare the independently frozen 30-point sphere `1->4` fit downstream. | CLI |
| [other_scripts/sphere_five_point_30point_worldsheet_fit.py](code/tree_level/five_point/other_scripts/sphere_five_point_30point_worldsheet_fit.py) | Merge and fit the 30-point sphere `1->4` worldsheet campaign. | CLI |
| [other_scripts/sphere_five_point_continuation_checks.py](code/tree_level/five_point/other_scripts/sphere_five_point_continuation_checks.py) | Low-cost checks for the first sphere-five Liouville contour crossing. | Script |
| [other_scripts/sphere_five_point_imaginary_ray_fit.py](code/tree_level/five_point/other_scripts/sphere_five_point_imaginary_ray_fit.py) | Freeze a target-free quadratic fit to the sphere 1->4 imaginary-ray data. | CLI |
| [other_scripts/sphere_five_point_liouville_checks.py](code/tree_level/five_point/other_scripts/sphere_five_point_liouville_checks.py) | Geometry and low-cost numerical checks for the five-point Liouville atlas. | Script |

</details>

<details>
<summary>Sphere 1→5 — 42 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [conformal_block/six_point_arbitrary_precision_c_recursion.py](code/tree_level/six_point/conformal_block/six_point_arbitrary_precision_c_recursion.py) | Evaluate fixed-weight six-point c-recursion using arbitrary-precision scalar arithmetic. | Library |
| [conformal_block/six_point_block_coefficient_provider.py](code/tree_level/six_point/conformal_block/six_point_block_coefficient_provider.py) | Construct and verify the six-point CCY coefficients used by the momentum-bank builders. | Library |
| [conformal_block/six_point_collar_block_evaluator.py](code/tree_level/six_point/conformal_block/six_point_collar_block_evaluator.py) | Evaluate same-channel collar blocks with complete middle-nome coefficients. | Library |
| [conformal_block/six_point_measure_block_convergence.py](code/tree_level/six_point/conformal_block/six_point_measure_block_convergence.py) | Compare finite prefixes of the native six-point block on identical nodes. | Library |
| [conformal_block/six_point_refinement_statistics.py](code/tree_level/six_point/conformal_block/six_point_refinement_statistics.py) | Selective block refinement, immutable records, and paired Monte Carlo statistics. | Library |
| [conformal_block/six_point_select_integration_channels.py](code/tree_level/six_point/conformal_block/six_point_select_integration_channels.py) | Choose a six-point channel while preserving the original integration measure. | Library |
| [liouville_integration/six_point_analyze_quadrature_and_order_controls.py](code/tree_level/six_point/liouville_integration/six_point_analyze_quadrature_and_order_controls.py) | Summarize saved common-node momentum-rule and block-order comparisons. | CLI |
| [liouville_integration/six_point_build_momentum_banks.py](code/tree_level/six_point/liouville_integration/six_point_build_momentum_banks.py) | Build disjoint coefficient-bank shards and assemble them after checking complete node coverage. | CLI |
| [liouville_integration/six_point_collar_momentum_quadrature.py](code/tree_level/six_point/liouville_integration/six_point_collar_momentum_quadrature.py) | Momentum proposals and full DOZZ data for the bounded analytic-collar pilot. | Library |
| [liouville_integration/six_point_compare_momentum_quadratures.py](code/tree_level/six_point/liouville_integration/six_point_compare_momentum_quadratures.py) | Compare specified Liouville momentum rules on the same saved sphere geometry. | CLI |
| [liouville_integration/six_point_deduplicate_momentum_nodes.py](code/tree_level/six_point/liouville_integration/six_point_deduplicate_momentum_nodes.py) | Identify exact repeated momentum nodes so their CFT coefficients can be computed once. | CLI |
| [liouville_integration/six_point_endpoint_and_tail_quadrature.py](code/tree_level/six_point/liouville_integration/six_point_endpoint_and_tail_quadrature.py) | Explicit endpoint/core and infinite-tail proposals for sphere six points. | Library |
| [liouville_integration/six_point_mixed_order_node_catalog.py](code/tree_level/six_point/liouville_integration/six_point_mixed_order_node_catalog.py) | Merge bank manifests into a shared-node catalogue, build at the needed orders and assemble each view. | CLI |
| [liouville_integration/six_point_momentum_banks_and_reweighting.py](code/tree_level/six_point/liouville_integration/six_point_momentum_banks_and_reweighting.py) | Reusable, exactly reweighted momentum banks for sphere six-point CFTs. | Library |
| [liouville_integration/six_point_momentum_control_proposals.py](code/tree_level/six_point/liouville_integration/six_point_momentum_control_proposals.py) | Momentum proposals and one-axis controls used by the six-point curve. | Library |
| [liouville_integration/six_point_reuse_saved_coefficients.py](code/tree_level/six_point/liouville_integration/six_point_reuse_saved_coefficients.py) | Validate and reuse coefficient rows produced by another saved bank build. | CLI |
| [moduli_integration/six_point_analytic_collar_subtraction.py](code/tree_level/six_point/moduli_integration/six_point_analytic_collar_subtraction.py) | Analytic primary subtraction on an explicit labelled six-point collar. | Library |
| [moduli_integration/six_point_assemble_curve_amplitudes.py](code/tree_level/six_point/moduli_integration/six_point_assemble_curve_amplitudes.py) | Assemble audited per-energy receipts for the sphere 1->5 curve. | CLI |
| [moduli_integration/six_point_assemble_radial_curve_covariance.py](code/tree_level/six_point/moduli_integration/six_point_assemble_radial_curve_covariance.py) | Aggregate the fifteen-energy curve and retain covariance from shared scrambles. | CLI |
| [moduli_integration/six_point_batch_block_contraction.py](code/tree_level/six_point/moduli_integration/six_point_batch_block_contraction.py) | Shared finite-sum contractions for sphere six-point coefficient banks. | Library |
| [moduli_integration/six_point_boundary_strata.py](code/tree_level/six_point/moduli_integration/six_point_boundary_strata.py) | Complete labelled boundary combinatorics and source-frame collar diagnostics. | Library |
| [moduli_integration/six_point_channel_coordinates.py](code/tree_level/six_point/moduli_integration/six_point_channel_coordinates.py) | Linear-channel plumbing atlas for the six-punctured sphere. | Library |
| [moduli_integration/six_point_check_sampling_variance.py](code/tree_level/six_point/moduli_integration/six_point_check_sampling_variance.py) | Conservative generic radial second-moment range for the sphere 1->5 chamber. | Library |
| [moduli_integration/six_point_collar_counterterm_geometry.py](code/tree_level/six_point/moduli_integration/six_point_collar_counterterm_geometry.py) | Fixed labelled collar counterterms in the unchanged sampled sphere measure. | Library |
| [moduli_integration/six_point_combine_truncation_orders.py](code/tree_level/six_point/moduli_integration/six_point_combine_truncation_orders.py) | Combine two block truncation levels with explicit records of the available control samples. | Library |
| [moduli_integration/six_point_estimate_collar_corrected_integrand.py](code/tree_level/six_point/moduli_integration/six_point_estimate_collar_corrected_integrand.py) | Combine named coefficient banks with boundary markings and collar addback terms. | Library |
| [moduli_integration/six_point_evaluate_routed_geometry_shards.py](code/tree_level/six_point/moduli_integration/six_point_evaluate_routed_geometry_shards.py) | Evaluate indexed geometry shards with the specified channel and block representation. | Library |
| [moduli_integration/six_point_integrate_collar_patch.py](code/tree_level/six_point/moduli_integration/six_point_integrate_collar_patch.py) | Bounded labelled-patch CFT pilot for analytic two-outer-collar subtraction. | CLI |
| [moduli_integration/six_point_integrate_mixed_order_amplitudes.py](code/tree_level/six_point/moduli_integration/six_point_integrate_mixed_order_amplitudes.py) | Integrate a complete sphere sampling design with explicitly supplied mixed block orders. | CLI |
| [moduli_integration/six_point_prepare_radial_geometry.py](code/tree_level/six_point/moduli_integration/six_point_prepare_radial_geometry.py) | Prepare audited face-stratified six-point geometry and old-data reuse maps. | CLI |
| [moduli_integration/six_point_radial_sampling_distribution.py](code/tree_level/six_point/moduli_integration/six_point_radial_sampling_distribution.py) | Radial source proposals for the sphere six-point atlas. | Library |
| [moduli_integration/six_point_refinement_assembly.py](code/tree_level/six_point/moduli_integration/six_point_refinement_assembly.py) | Apply selected-node higher-order corrections and assemble the C14/C18 amplitude results. | CLI |
| [moduli_integration/six_point_route_collar_contributions.py](code/tree_level/six_point/moduli_integration/six_point_route_collar_contributions.py) | Route collar CFT contributions through explicitly marked channels. | Library |
| [moduli_integration/six_point_sample_integration_geometry.py](code/tree_level/six_point/moduli_integration/six_point_sample_integration_geometry.py) | Generate reusable, energy-independent geometry for the six-point sampling mixture. | Library |
| [moduli_integration/six_point_stable_puncture_coordinates.py](code/tree_level/six_point/moduli_integration/six_point_stable_puncture_coordinates.py) | Construct six-sphere coordinates using factorized puncture differences for numerical stability. | Library |
| [production/dispatch_shared_refinement_queue.py](code/tree_level/six_point/production/dispatch_shared_refinement_queue.py) | Distribute unfinished six-point refinement batches across the configured shared worker queue. | CLI |
| [production/six_point_prepare_amplitude_curve.py](code/tree_level/six_point/production/six_point_prepare_amplitude_curve.py) | Prepare per-energy runs from an accepted radial production release and its coefficient banks. | CLI |
| [production/six_point_refinement.py](code/tree_level/six_point/production/six_point_refinement.py) | Prepare, evaluate, and assemble selective C13-C14 and C15-C18 refinement stages. | CLI |
| [production/six_point_run_frozen_energy_worker.py](code/tree_level/six_point/production/six_point_run_frozen_energy_worker.py) | Evaluate one prepared sphere 1→5 energy release with the retained worker configuration. | CLI |
| [production/six_point_run_frozen_radial_worker.py](code/tree_level/six_point/production/six_point_run_frozen_radial_worker.py) | Evaluate the prepared radial sampling design and reuse validated completed records. | CLI |
| [production/six_point_submit_radial_campaign.py](code/tree_level/six_point/production/six_point_submit_radial_campaign.py) | Submit a prepared radial campaign to the cluster and record the submission. | CLI |
| [production/verify_saved_c14_continuation_inputs.py](code/tree_level/six_point/production/verify_saved_c14_continuation_inputs.py) | Verify saved C14 continuation inputs by hashes and bookkeeping without replaying integrals. | Library |

</details>

<details>
<summary>Torus 1→1 — 13 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [conformal_block/torus_two_point_blocks.py](code/one_loop/two_point/conformal_block/torus_two_point_blocks.py) | Direct and recursive Virasoro torus two-point blocks in two channels. | Library |
| [liouville_integration/genus1_two_point_coefficient_bank.py](code/one_loop/two_point/liouville_integration/genus1_two_point_coefficient_bank.py) | Prepare and preserve two-point Liouville coefficients for thermal reuse. | CLI |
| [liouville_integration/genus1_two_point_worldsheet.py](code/one_loop/two_point/liouville_integration/genus1_two_point_worldsheet.py) | Numerical worldsheet ingredients for the c=1 genus-one two-point amplitude. | Library |
| [moduli_integration/genus1_two_point_thermal_integration.py](code/one_loop/two_point/moduli_integration/genus1_two_point_thermal_integration.py) | Thermal 1->1 worldsheet integral with an explicit compact-boson lattice. | CLI |
| [moduli_integration/integrate_genus1_two_point_worldsheet.py](code/one_loop/two_point/moduli_integration/integrate_genus1_two_point_worldsheet.py) | RQMC integration of the genus-one c=1 two-point worldsheet amplitude. | CLI |
| [other_scripts/checks/genus1_two_point_worldsheet_checks.py](code/one_loop/two_point/other_scripts/checks/genus1_two_point_worldsheet_checks.py) | Pointwise and channel-overlap checks for the genus-one worldsheet code. | CLI |
| [other_scripts/checks/torus_necklace_blocks_checks.py](code/one_loop/two_point/other_scripts/checks/torus_necklace_blocks_checks.py) | Low-level checks for torus two- and three-point h-recursion blocks. | Script |
| [other_scripts/checks/torus_two_point_blocks_checks.py](code/one_loop/two_point/other_scripts/checks/torus_two_point_blocks_checks.py) | Checks for native-q torus two-point c recursion and its independent baseline. | Script |
| [other_scripts/fit_genus1_two_point_bry_scan.py](code/one_loop/two_point/other_scripts/fit_genus1_two_point_bry_scan.py) | Post-freeze BRY fit of the genus-one two-point imaginary-energy scan. | CLI |
| [other_scripts/genus1_two_point_thermal_production.py](code/one_loop/two_point/other_scripts/genus1_two_point_thermal_production.py) | Prepare coefficient banks and run the thermal genus-one 1->1 radius scan. | CLI |
| [other_scripts/genus1_two_point_thermal_report.py](code/one_loop/two_point/other_scripts/genus1_two_point_thermal_report.py) | Assemble thermal 1->1 receipts and optionally render the draft-style figure. | CLI |
| [other_scripts/plot_genus1_two_point_hybrid_recursion.py](code/one_loop/two_point/other_scripts/plot_genus1_two_point_hybrid_recursion.py) | Plot the hybrid h/c-recursive genus-one 1->1 amplitude scan as SVG. | CLI |
| [other_scripts/run_genus1_two_point_imaginary_scan.py](code/one_loop/two_point/other_scripts/run_genus1_two_point_imaginary_scan.py) | Run the frozen ten-point imaginary-energy worldsheet smoke design. | CLI |

</details>

<details>
<summary>Torus 1→2, including thermal kernels — 61 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [conformal_block/genus1_collision_elliptic.py](code/one_loop/three_point/conformal_block/genus1_collision_elliptic.py) | Explicit, restartable preparation of same-channel collision H coefficients. | Library |
| [conformal_block/genus1_native_log_evaluation.py](code/one_loop/three_point/conformal_block/genus1_native_log_evaluation.py) | Evaluate native-q block polynomials in a scaled logarithmic form to avoid overflow. | Library |
| [conformal_block/torus_c_recursion_numerics.py](code/one_loop/three_point/conformal_block/torus_c_recursion_numerics.py) | Verified fixed-weight CCY c recursion for torus sewing graphs. | Library |
| [conformal_block/torus_three_point_blocks.py](code/one_loop/three_point/conformal_block/torus_three_point_blocks.py) | Native necklace, pair, and linear-channel torus three-point blocks. | Library |
| [liouville_integration/genus1_boundary_banks.py](code/one_loop/three_point/liouville_integration/genus1_boundary_banks.py) | Verified native and elliptic five-point bank adapters for boundary cubature. | Library |
| [liouville_integration/genus1_boundary_spectrum.py](code/one_loop/three_point/liouville_integration/genus1_boundary_spectrum.py) | Kernel-specific spectral inputs for the staged amplitude controller. | Library |
| [liouville_integration/genus1_channel_evaluation.py](code/one_loop/three_point/liouville_integration/genus1_channel_evaluation.py) | Evaluate channel contributions using verified coefficient banks and the chosen moduli-sector convention. | Library |
| [liouville_integration/genus1_collar_mp_arithmetic.py](code/one_loop/three_point/liouville_integration/genus1_collar_mp_arithmetic.py) | Multiprecision evaluation of unchanged finite collar polynomials. | Library |
| [liouville_integration/genus1_native_bank.py](code/one_loop/three_point/liouville_integration/genus1_native_bank.py) | Preparation, integrity checks, and evaluation of saved native block banks. | Library |
| [liouville_integration/genus1_projection_momentum.py](code/one_loop/three_point/liouville_integration/genus1_projection_momentum.py) | Construct Liouville momentum rules for boundary projections at specified geometries. | Library |
| [liouville_integration/genus1_projection_preparation.py](code/one_loop/three_point/liouville_integration/genus1_projection_preparation.py) | Explicit prepare/build/evaluate path for kernel-specific Liouville grids. | CLI |
| [liouville_integration/genus1_three_point_momentum.py](code/one_loop/three_point/liouville_integration/genus1_three_point_momentum.py) | Geometry-dependent, refinement-gated Liouville momentum integration. | Library |
| [liouville_integration/genus1_three_point_worldsheet.py](code/one_loop/three_point/liouville_integration/genus1_three_point_worldsheet.py) | Blind worldsheet kernel for the c=1 genus-one 1->2 amplitude. | Library |
| [liouville_integration/thermal/banks.py](code/one_loop/three_point/liouville_integration/thermal/banks.py) | Prepare and build thermal Liouville coefficient banks with the resumable CFT builder. | Library |
| [liouville_integration/thermal/evaluation.py](code/one_loop/three_point/liouville_integration/thermal/evaluation.py) | Read and evaluate verified thermal coefficient banks. | Library |
| [liouville_integration/thermal/paired_quadrature.py](code/one_loop/three_point/liouville_integration/thermal/paired_quadrature.py) | Place thermal momentum nodes using the sampled geometry at a twelve-node budget. | Script |
| [liouville_integration/thermal/quadrature.py](code/one_loop/three_point/liouville_integration/thermal/quadrature.py) | Construct momentum rules for the two thermally modified integrated endpoints. | Library |
| [liouville_integration/thermal/thermal_cut_rule.py](code/one_loop/three_point/liouville_integration/thermal/thermal_cut_rule.py) | Thermal-kernel-aware cut-momentum rule for preparing FUTURE CFT banks. | Script |
| [liouville_integration/thermal/thermal_saved_inputs.py](code/one_loop/three_point/liouville_integration/thermal/thermal_saved_inputs.py) | Verified saved Liouville inputs and fixed thermal sampling conventions. | Library |
| [moduli_integration/genus1_analytic_boundary_layer.py](code/one_loop/three_point/moduli_integration/genus1_analytic_boundary_layer.py) | Opt-in analytic boundary partition and direct compact-bulk sampling. | Library |
| [moduli_integration/genus1_boundary_batch.py](code/one_loop/three_point/moduli_integration/genus1_boundary_batch.py) | Reuse geometry and finite elliptic-block contractions across boundary evaluations. | Library |
| [moduli_integration/genus1_boundary_cubature.py](code/one_loop/three_point/moduli_integration/genus1_boundary_cubature.py) | Adapt deterministic moduli quadrature while caching completed boundary evaluations. | Library |
| [moduli_integration/genus1_boundary_integration.py](code/one_loop/three_point/moduli_integration/genus1_boundary_integration.py) | Drive deterministic collision-region integration using the analytic boundary decomposition. | Library |
| [moduli_integration/genus1_collision_sampling.py](code/one_loop/three_point/moduli_integration/genus1_collision_sampling.py) | Collision-region maps, physical weights, and requests to the Liouville evaluator. | Library |
| [moduli_integration/genus1_comb_collar.py](code/one_loop/three_point/moduli_integration/genus1_comb_collar.py) | Evaluate triple-collision subtraction terms and analytic faces in comb coordinates. | Library |
| [moduli_integration/genus1_cusp_control.py](code/one_loop/three_point/moduli_integration/genus1_cusp_control.py) | Evaluate a matched CFT control term on the compact cusp integration domain. | Library |
| [moduli_integration/genus1_integrated_scale_collar.py](code/one_loop/three_point/moduli_integration/genus1_integrated_scale_collar.py) | Integrate the collision scale using saved coefficient banks and analytic collar formulas. | Library |
| [moduli_integration/genus1_modular_geometry.py](code/one_loop/three_point/moduli_integration/genus1_modular_geometry.py) | Marked torus frames and pullbacks to the original integration coordinates. | Library |
| [moduli_integration/genus1_moduli_panels.py](code/one_loop/three_point/moduli_integration/genus1_moduli_panels.py) | Opt-in, anisotropic moduli panels for the retained analytic estimator. | Library |
| [moduli_integration/genus1_moduli_sampling.py](code/one_loop/three_point/moduli_integration/genus1_moduli_sampling.py) | Torus and nonseparating coordinates, sampling, and Monte Carlo summaries. | Library |
| [moduli_integration/genus1_nonseparating.py](code/one_loop/three_point/moduli_integration/genus1_nonseparating.py) | Evaluate the analytic long-tube contribution using verified native CFT banks. | Library |
| [moduli_integration/genus1_pair_collar.py](code/one_loop/three_point/moduli_integration/genus1_pair_collar.py) | Analytic leading pair-collar projection using saved native-q banks. | Library |
| [moduli_integration/genus1_sewing_domains.py](code/one_loop/three_point/moduli_integration/genus1_sewing_domains.py) | Conservative radial-contour domains for native torus sewing series. | Library |
| [moduli_integration/genus1_three_point_channel_atlas.py](code/one_loop/three_point/moduli_integration/genus1_three_point_channel_atlas.py) | Channel atlas for the c=1 genus-one three-point worldsheet integrand. | Library |
| [moduli_integration/genus1_three_point_collision.py](code/one_loop/three_point/moduli_integration/genus1_three_point_collision.py) | Collision-resolved integrands for the torus three-point amplitude. | Library |
| [moduli_integration/thermal/kernels.py](code/one_loop/three_point/moduli_integration/thermal/kernels.py) | Evaluate thermal kernels with explicit Kaluza-Klein charges. | Script |
| [moduli_integration/thermal/radius_scan.py](code/one_loop/three_point/moduli_integration/thermal/radius_scan.py) | Radius-specific moduli integration of the retained genus-one estimator. | CLI |
| [moduli_integration/thermal/remainders.py](code/one_loop/three_point/moduli_integration/thermal/remainders.py) | Evaluate thermal collision remainders at common CFT momentum nodes. | Library |
| [moduli_integration/thermal/study_pair_sampling.py](code/one_loop/three_point/moduli_integration/thermal/study_pair_sampling.py) | Independent moduli-sampling experiment for the thermal pair-collision term. | CLI |
| [other_scripts/checks/genus1_three_point_c_recursion_checks.py](code/one_loop/three_point/other_scripts/checks/genus1_three_point_c_recursion_checks.py) | PBW anchors for the c-only genus-one three-point production backend. | CLI |
| [other_scripts/genus1_amplitude_curve.py](code/one_loop/three_point/other_scripts/genus1_amplitude_curve.py) | Prepare and evaluate a fixed energy-grid design with explicit worker allocation. | CLI |
| [other_scripts/genus1_amplitude_report.py](code/one_loop/three_point/other_scripts/genus1_amplitude_report.py) | Amplitude assembly diagnostics and numerical error reports. | CLI |
| [other_scripts/genus1_compact_replay.py](code/one_loop/three_point/other_scripts/genus1_compact_replay.py) | Replay the existing compact observations; never draw new MC points. | CLI |
| [other_scripts/genus1_final_curve_plan.py](code/one_loop/three_point/other_scripts/genus1_final_curve_plan.py) | Prepare the twenty-energy curve from a previous curve and pair-cell refinement, reusing completed work. | CLI |
| [other_scripts/genus1_fresh_analytic.py](code/one_loop/three_point/other_scripts/genus1_fresh_analytic.py) | Prepare and run bounded calculations with the analytic boundary decomposition. | CLI |
| [other_scripts/genus1_moduli_amplitude.py](code/one_loop/three_point/other_scripts/genus1_moduli_amplitude.py) | Prepare, evaluate and reduce full-domain moduli samples using a specified CFT estimator. | CLI |
| [other_scripts/genus1_moduli_panel_campaign.py](code/one_loop/three_point/other_scripts/genus1_moduli_panel_campaign.py) | Split moduli-panel calculations into cost-balanced jobs and collect their results. | CLI |
| [other_scripts/genus1_moduli_stability.py](code/one_loop/three_point/other_scripts/genus1_moduli_stability.py) | Compare quadrature choices at fixed amplitude inputs and selected moduli regions. | CLI |
| [other_scripts/genus1_moduli_stability_curve.py](code/one_loop/three_point/other_scripts/genus1_moduli_stability_curve.py) | Apply the selected quadrature stability comparison across saved energies. | CLI |
| [other_scripts/genus1_pair_cell_refinement.py](code/one_loop/three_point/other_scripts/genus1_pair_cell_refinement.py) | Replace one saved pair-collision cell with a refined estimate and retain the other components. | CLI |
| [other_scripts/genus1_primary_first_production.py](code/one_loop/three_point/other_scripts/genus1_primary_first_production.py) | Assemble a production amplitude using tested primary angular panels and saved contributions. | CLI |
| [other_scripts/genus1_primary_moduli_refinement.py](code/one_loop/three_point/other_scripts/genus1_primary_moduli_refinement.py) | Check physical-length and angular quadrature in selected primary cells. | CLI |
| [other_scripts/genus1_production_support.py](code/one_loop/three_point/other_scripts/genus1_production_support.py) | Production configuration, checked records, and worker resource limits. | Library |
| [other_scripts/genus1_rebuild_policy.py](code/one_loop/three_point/other_scripts/genus1_rebuild_policy.py) | Define acceptance criteria for the corrected genus-one frequency-grid rebuild. | Script |
| [other_scripts/genus1_source_paths.py](code/one_loop/three_point/other_scripts/genus1_source_paths.py) | Install the retained genus-one import paths and resolve production directories. | Library |
| [other_scripts/genus1_upper_pair_refinement.py](code/one_loop/three_point/other_scripts/genus1_upper_pair_refinement.py) | Compare coarse and fine physical panels for one upper pair-collision cell. | CLI |
| [other_scripts/plot_genus1_production.py](code/one_loop/three_point/other_scripts/plot_genus1_production.py) | Draw a historical torus amplitude curve with grid-RMS bars and a matrix-model comparison. | Script |
| [other_scripts/thermal/draft_radius_campaign.py](code/one_loop/three_point/other_scripts/thermal/draft_radius_campaign.py) | Plan and evaluate paired thermal corrections at the draft radii using prepared banks. | CLI |
| [other_scripts/thermal/draft_radius_collect.py](code/one_loop/three_point/other_scripts/thermal/draft_radius_collect.py) | Combine the saved zero-temperature baseline with completed thermal corrections. | CLI |
| [other_scripts/thermal/draft_radius_validation.py](code/one_loop/three_point/other_scripts/thermal/draft_radius_validation.py) | Compare eight- and twelve-node thermal momentum rules at fixed moduli. | CLI |
| [other_scripts/thermal/reference_campaign.py](code/one_loop/three_point/other_scripts/thermal/reference_campaign.py) | Build thermal reference banks within a twelve-node budget and inspect earlier results. | CLI |

</details>

<details>
<summary>Genus-two free energy and channel matching — 45 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [conformal_block/ccy_genus2_glasses_block.py](code/two_loop/conformal_block/ccy_genus2_glasses_block.py) | CCY c-recursion for the genus-two separating/glasses pants graph. | CLI |
| [geometry/audit_q_to_omega_accuracy.py](code/two_loop/geometry/audit_q_to_omega_accuracy.py) | Audit plumbing coordinates against their target period matrices. | CLI |
| [geometry/bolza_ccy_recursion.py](code/two_loop/geometry/bolza_ccy_recursion.py) | Evaluate the CCY Liouville genus-two recursion near the Bolza point. | CLI |
| [geometry/bolza_torus_plumbing_reach.py](code/two_loop/geometry/bolza_torus_plumbing_reach.py) | Search for a two-torus plumbing chart reaching the Bolza curve. | CLI |
| [geometry/genus2_calibrated_schottky.py](code/two_loop/geometry/genus2_calibrated_schottky.py) | Guarded Schottky period evaluation for calibrated regions only. | Library |
| [geometry/genus2_holomorphic_period_table.py](code/two_loop/geometry/genus2_holomorphic_period_table.py) | Reference table and calibrated Schottky envelope for the genus-two period map. | Library |
| [geometry/genus2_hybrid_period_map.py](code/two_loop/geometry/genus2_hybrid_period_map.py) | Adaptive full-chart genus-two `q -> Omega` period-map policy. | Library |
| [geometry/genus2_integration_geometry.py](code/two_loop/geometry/genus2_integration_geometry.py) | Plumbing-chart selection, period-map checks, and saved coordinate reconstruction. | Library |
| [geometry/genus2_multiprecision_collocation.py](code/two_loop/geometry/genus2_multiprecision_collocation.py) | Certified rescaled holomorphic-form solver for mixed plumbing cusps. | Library |
| [geometry/genus2_period_table.py](code/two_loop/geometry/genus2_period_table.py) | Indexed mixed-backend genus-two period table. | Library |
| [geometry/genus2_period_table_grid.py](code/two_loop/geometry/genus2_period_table_grid.py) | Prepare deterministic genus-two period-table manifests. | CLI |
| [geometry/genus2_period_table_selector.py](code/two_loop/geometry/genus2_period_table_selector.py) | Atlas-aware q selector for the genus-two period table. | Library |
| [geometry/genus2_plumbing_atlas.py](code/two_loop/geometry/genus2_plumbing_atlas.py) | Mapping-class atlas diagnostics for genus-two plumbing coordinates. | CLI |
| [geometry/liouville_genus2_modular_check.py](code/two_loop/geometry/liouville_genus2_modular_check.py) | Modular-covariance diagnostics for the genus-two Liouville plumbing result. | CLI |
| [liouville_integration/genus2_liouville_evaluation.py](code/two_loop/liouville_integration/genus2_liouville_evaluation.py) | Liouville integration in the theta and glasses channels, including logarithmic scaling. | Library |
| [liouville_integration/liouville_genus2.py](code/two_loop/liouville_integration/liouville_genus2.py) | Genus-two Liouville partition function in the pair-of-tori plumbing channel. | CLI |
| [liouville_integration/liouville_genus2_ccy.py](code/two_loop/liouville_integration/liouville_genus2_ccy.py) | Raw Liouville genus-two plumbing integral from the CCY block. | CLI |
| [liouville_integration/liouville_genus2_glasses.py](code/two_loop/liouville_integration/liouville_genus2_glasses.py) | Raw Liouville genus-two plumbing integral in the glasses frame. | CLI |
| [moduli_integration/genus2_bc_ghost.py](code/two_loop/moduli_integration/genus2_bc_ghost.py) | Canonical genus-two `bc` ghost density in period-matrix coordinates. | CLI |
| [moduli_integration/genus2_c1_string_integrand.py](code/two_loop/moduli_integration/genus2_c1_string_integrand.py) | Genus-two `c=1` string integrand in period-matrix coordinates. | Library |
| [moduli_integration/genus2_moduli_estimation.py](code/two_loop/moduli_integration/genus2_moduli_estimation.py) | Saved sampling weights, normalization conventions, and moduli-integral estimates. | Library |
| [moduli_integration/genus2_moduli_sampling.py](code/two_loop/moduli_integration/genus2_moduli_sampling.py) | Invariant-measure and physical-mixture Sobol designs, with explicit sampling weights. | CLI |
| [moduli_integration/genus2_node_evaluation.py](code/two_loop/moduli_integration/genus2_node_evaluation.py) | Evaluate the full genus-two integrand at one saved moduli sampling point. | Library |
| [moduli_integration/genus2_scalar_evaluation.py](code/two_loop/moduli_integration/genus2_scalar_evaluation.py) | Scalar partition functions and their truncation checks in a fixed plumbing frame. | Library |
| [moduli_integration/genus2_separating_tail.py](code/two_loop/moduli_integration/genus2_separating_tail.py) | Separating-primary control variate for the current c=1 genus-two kernel. | Library |
| [moduli_integration/genus2_siegel_fundamental_domain.py](code/two_loop/moduli_integration/genus2_siegel_fundamental_domain.py) | Genus-two Gottschling domain and invariant Siegel-volume sampling. | CLI |
| [other_scripts/assemble_genus2_resolved_production.py](code/two_loop/other_scripts/assemble_genus2_resolved_production.py) | Combine resolved genus-two bulk and separating data in the absolute BRY coupling convention. | CLI |
| [other_scripts/assemble_genus2_tail_completion.py](code/two_loop/other_scripts/assemble_genus2_tail_completion.py) | Assemble measured separating-tail additions while retaining unresolved-error information. | CLI |
| [other_scripts/audit_genus2_g_convention.py](code/two_loop/other_scripts/audit_genus2_g_convention.py) | Audit the frozen BRY-v7 result and export it in g=mu^-1=2*pi*g_s units. | Script |
| [other_scripts/check_channel_convergence.py](code/two_loop/other_scripts/check_channel_convergence.py) | Re-evaluate one Liouville integral at fixed saved geometry/scalar normalization with chosen block and momentum orders. | CLI |
| [other_scripts/checks/ccy_genus2_glasses_block_checks.py](code/two_loop/other_scripts/checks/ccy_genus2_glasses_block_checks.py) | Checks for the CCY genus-two glasses-frame recursion. | Script |
| [other_scripts/checks/genus2_c1_string_integrand_checks.py](code/two_loop/other_scripts/checks/genus2_c1_string_integrand_checks.py) | Checks for the explicit genus-two `c=1` string integrand. | Script |
| [other_scripts/checks/genus2_separating_tail_checks.py](code/two_loop/other_scripts/checks/genus2_separating_tail_checks.py) | Independent identities and optional full-CFT overlap checks for the tail. | CLI |
| [other_scripts/checks/genus2_tail_completion_checks.py](code/two_loop/other_scripts/checks/genus2_tail_completion_checks.py) | Saved-node reconstruction, covariance, and independent torus-block checks. | Script |
| [other_scripts/checks/genus2_tail_precision_checks.py](code/two_loop/other_scripts/checks/genus2_tail_precision_checks.py) | Check the arithmetic fixes for the two retained separating-tail failure cases. | Script |
| [other_scripts/checks/plot_genus2_absolute_bry_checks.py](code/two_loop/other_scripts/checks/plot_genus2_absolute_bry_checks.py) | Fit/covariance regressions and frozen-data normalization round trip. | Script |
| [other_scripts/genus2_glasses_production_gate.py](code/two_loop/other_scripts/genus2_glasses_production_gate.py) | Check precision and period sensitivity for selected glasses-channel production points. | CLI |
| [other_scripts/genus2_moduli_production.py](code/two_loop/other_scripts/genus2_moduli_production.py) | Pilot Monte Carlo integration of the genus-two c=1 string density. | CLI |
| [other_scripts/genus2_separating_production.py](code/two_loop/other_scripts/genus2_separating_production.py) | Genus-two separating-region production, including frozen-node validation and order controls. | CLI |
| [other_scripts/genus2_separating_residual_retry.py](code/two_loop/other_scripts/genus2_separating_residual_retry.py) | Re-evaluate one separating-region residual at higher momentum resolution. | CLI |
| [other_scripts/plot_genus2_absolute_bry.py](code/two_loop/other_scripts/plot_genus2_absolute_bry.py) | Plot the saved absolute genus-two free energy and fit it with correlated errors. | CLI |
| [other_scripts/rebuild_genus2_bulk_control_groups.py](code/two_loop/other_scripts/rebuild_genus2_bulk_control_groups.py) | Reconstruct bulk/control groups from the saved nodes and radius cache. | CLI |
| [other_scripts/recompute_channel_comparison.py](code/two_loop/other_scripts/recompute_channel_comparison.py) | Solve target geometries, compute scalar and Liouville factors, and assemble the N12/n12 baseline. | CLI |
| [other_scripts/recover_channel_convergence.py](code/two_loop/other_scripts/recover_channel_convergence.py) | Verify completed fixed-geometry receipts and export convergence comparisons and the current N12/n16 table. | CLI |
| [other_scripts/study_genus2_separating_tail_correction.py](code/two_loop/other_scripts/study_genus2_separating_tail_correction.py) | Apply a separating-tail correction to a saved mixed-channel run and report its effect. | CLI |

</details>

<details>
<summary>Figure programs — 5 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [export_current_convention_results.py](code/figures/export_current_convention_results.py) | Re-export saved amplitudes and the genus-two free energy in the current coupling convention; expects original campaign inputs. | CLI |
| [plot_recovered_results.py](code/figures/plot_recovered_results.py) | Render the eight draft panels using only newly recovered numerical tables. | Library |
| [plot_thermal_one_loop_radius.py](code/figures/plot_thermal_one_loop_radius.py) | Format historical thermal torus scans for plotting; uses fixed campaign paths. | Script |
| [plot_thermal_tree_radius.py](code/figures/plot_thermal_tree_radius.py) | Convert saved tree amplitudes to radius curves using R=1/t and draw the comparison plots. | CLI |
| [replot_sphere_six_direct_series.py](code/figures/replot_sphere_six_direct_series.py) | Assemble and plot the saved direct six-point series from its original campaign records. | CLI |

</details>

<details>
<summary>Vendored genus-zero elliptic recursion — 6 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [__init__.py](code/general/virasoro/vendor/genus0_elliptic_h_recursion/src/genus0_elliptic_h_recursion/__init__.py) | Genus-zero Virasoro n-point elliptic h-recursion. | Library |
| [__main__.py](code/general/virasoro/vendor/genus0_elliptic_h_recursion/src/genus0_elliptic_h_recursion/__main__.py) | Run a compact six-point demonstration with `python -m`. | Package demo |
| [block.py](code/general/virasoro/vendor/genus0_elliptic_h_recursion/src/genus0_elliptic_h_recursion/block.py) | Full sphere-block reconstruction from the reduced elliptic recursion. | Library |
| [geometry.py](code/general/virasoro/vendor/genus0_elliptic_h_recursion/src/genus0_elliptic_h_recursion/geometry.py) | Pillow coordinates for the aligned real genus-zero comb cell. | Library |
| [kac.py](code/general/virasoro/vendor/genus0_elliptic_h_recursion/src/genus0_elliptic_h_recursion/kac.py) | Kac weights, Zamolodchikov residues, and fusion polynomials. | Library |
| [recursion.py](code/general/virasoro/vendor/genus0_elliptic_h_recursion/src/genus0_elliptic_h_recursion/recursion.py) | General fixed-difference elliptic h-recursion for sphere comb blocks. | Library |

</details>

### Preserved evaluation drivers

The three Python files below are immutable copies of drivers used for the
retained channel computations. They establish provenance and are not the
current execution entry points. Use the corresponding programs under
`code/two_loop/other_scripts/` for new work.

| Saved source | Purpose |
| --- | --- |
| [Baseline driver `fd26b1bb…`](data/channel_comparison/source_manifests/drivers/fd26b1bb755f9025e5ab096b1ad7900b52112a6700319b735803a1c09a071bcb.py) | Retained baseline evaluation driver, identified by its source hash. |
| [Baseline driver `c22b37e6…`](data/channel_comparison/source_manifests/drivers/c22b37e6a55216cb6ba1e12a70b6fe9c6f6413bfeaf0ebddf8f30c34c49212ee.py) | Additional retained baseline driver version, identified by its source hash. |
| [Convergence driver](data/channel_convergence/driver.py) | Exact fixed-geometry Liouville driver used for the completed convergence campaign. |
