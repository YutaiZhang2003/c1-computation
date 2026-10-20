# Command and file reference

This companion to the [README](README.md) provides detailed execution examples,
production input requirements and a purpose/interface entry for every Python
file. The README follows Appendix B and explains how the numerical components
fit together.

**Contents:** [Setup](#setup-and-quick-start) · [Choose a program](#which-program-should-i-use)
· [Launcher and libraries](#using-the-numerical-codes) · [New calculations](#starting-new-calculations)
· [Usage issues](#common-usage-issues) · [Provenance](#lightweight-data-and-provenance)
· [File catalogue](#file-by-file-code-catalogue)

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
| [`recover_channel_convergence.py`](code/two_loop/other_scripts/recover_channel_convergence.py) | Recover the current channel table and completed convergence checks | Saved N12/n16 and N18/n12 receipts plus the baseline → full-precision CSV and verification JSON | Use the [channel recovery command](README.md#cross-channel-matching) |
| [`check_channel_convergence.py`](code/two_loop/other_scripts/check_channel_convergence.py) | Recompute one Liouville integral at a saved geometry and scalar normalization | Row, channel, block order and momentum order → one new JSON receipt | Use `--dry-run` to inspect a job; see [fresh channel evaluations](#fresh-channel-evaluations) |
| [`recompute_channel_comparison.py`](code/two_loop/other_scripts/recompute_channel_comparison.py) | Solve the target geometry, evaluate scalar and Liouville factors, or assemble the N12/n12 baseline | Target matrices → per-channel receipts; baseline receipts → table | `evaluate` and `assemble` subcommands; see [fresh channel evaluations](#fresh-channel-evaluations) |
| [`plot_recovered_results.py`](code/figures/plot_recovered_results.py) | Draw the eight panels from recovered numerical tables | Recovered output directory → `figures/*.pdf`, `*.png`, `panels.json` | Library called by `reproduce.py`; see [redrawing figures](#redrawing-figures) |
| [`build_light_dataset.py`](code/general/recovery/build_light_dataset.py) | Construct the compact dataset from the original production archive | Original `production_integrated/` → a new lightweight input directory | Maintainer command in [provenance](#lightweight-data-and-provenance) |
| [`refresh_source_manifest.py`](code/general/recovery/refresh_source_manifest.py) | Record an intentional source/documentation update and refresh source-index hashes | Current repository files → `SOURCE_MANIFEST.json` and source indexes | Maintainer command in [provenance](#lightweight-data-and-provenance) |

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

The expandable tables describe all **220 Python files under `code/`**. Sector
purposes follow the corresponding `CODE_INDEX.json` entries; shared tools are
described from their module documentation. The
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
| [conformal_block/five_point_liouville_correlator_and_atlas.py](code/tree_level/five_point/conformal_block/five_point_liouville_correlator_and_atlas.py) | Decomposition channels, coordinate frames, and channel selection | Library |
| [conformal_block/five_point_virasoro_block_recursion.py](code/tree_level/five_point/conformal_block/five_point_virasoro_block_recursion.py) | Five-point block from the $c$-recursion | Library |
| [liouville_integration/five_point_integrand_and_quadrature.py](code/tree_level/five_point/liouville_integration/five_point_integrand_and_quadrature.py) | Momentum quadrature, precomputed DOZZ factors and block coefficients, and crossed-pole residues | Library |
| [moduli_integration/five_point_moduli_integration.py](code/tree_level/five_point/moduli_integration/five_point_moduli_integration.py) | Sampling weight, Monte Carlo points, moduli integral, and sampling standard error | Library |
| [other_scripts/ccy_sphere_five_point_checks.py](code/tree_level/five_point/other_scripts/ccy_sphere_five_point_checks.py) | Independent checks of the conformal block | Script |
| [other_scripts/export_sphere_five_point_t_axis_plot_data.py](code/tree_level/five_point/other_scripts/export_sphere_five_point_t_axis_plot_data.py) | Amplitude values and uncertainty estimates used in the figure | CLI |
| [other_scripts/five_point_production.py](code/tree_level/five_point/other_scripts/five_point_production.py) | Amplitude scan and its frozen twelve-point extension | CLI |
| [other_scripts/sphere_five_point_30point_audit_summary.py](code/tree_level/five_point/other_scripts/sphere_five_point_30point_audit_summary.py) | Comparisons with saved higher-order evaluations and checks of the amplitude table | CLI |
| [other_scripts/sphere_five_point_30point_checks.py](code/tree_level/five_point/other_scripts/sphere_five_point_30point_checks.py) | Comparisons with saved higher-order evaluations and checks of the amplitude table | Script |
| [other_scripts/sphere_five_point_30point_matrix_comparison.py](code/tree_level/five_point/other_scripts/sphere_five_point_30point_matrix_comparison.py) | Comparison with the matrix-model amplitude | CLI |
| [other_scripts/sphere_five_point_30point_worldsheet_fit.py](code/tree_level/five_point/other_scripts/sphere_five_point_30point_worldsheet_fit.py) | Combining saved amplitudes and fitting their energy dependence | CLI |
| [other_scripts/sphere_five_point_continuation_checks.py](code/tree_level/five_point/other_scripts/sphere_five_point_continuation_checks.py) | Checks of momentum quadrature and contour residues | Script |
| [other_scripts/sphere_five_point_imaginary_ray_fit.py](code/tree_level/five_point/other_scripts/sphere_five_point_imaginary_ray_fit.py) | Combining saved amplitudes and fitting their energy dependence | CLI |
| [other_scripts/sphere_five_point_liouville_checks.py](code/tree_level/five_point/other_scripts/sphere_five_point_liouville_checks.py) | Checks of coordinate frames and Jacobians | Script |

</details>

<details>
<summary>Sphere 1→5 — 42 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [conformal_block/six_point_arbitrary_precision_c_recursion.py](code/tree_level/six_point/conformal_block/six_point_arbitrary_precision_c_recursion.py) | Six-point block from the $c$-recursion and checks of arithmetic precision | Library |
| [conformal_block/six_point_block_coefficient_provider.py](code/tree_level/six_point/conformal_block/six_point_block_coefficient_provider.py) | Six-point block from the $c$-recursion and checks of arithmetic precision | Library |
| [conformal_block/six_point_collar_block_evaluator.py](code/tree_level/six_point/conformal_block/six_point_collar_block_evaluator.py) | Four-point elliptic block and outer descendant corrections in $D_{\mathrm{sep}}$ | Library |
| [conformal_block/six_point_measure_block_convergence.py](code/tree_level/six_point/conformal_block/six_point_measure_block_convergence.py) | Assessment of block truncation | Library |
| [conformal_block/six_point_refinement_statistics.py](code/tree_level/six_point/conformal_block/six_point_refinement_statistics.py) | Refinement criteria and paired Monte Carlo statistics | Library |
| [conformal_block/six_point_select_integration_channels.py](code/tree_level/six_point/conformal_block/six_point_select_integration_channels.py) | Channel selection | Library |
| [liouville_integration/six_point_analyze_quadrature_and_order_controls.py](code/tree_level/six_point/liouville_integration/six_point_analyze_quadrature_and_order_controls.py) | Assessment of momentum integration and block truncation | CLI |
| [liouville_integration/six_point_build_momentum_banks.py](code/tree_level/six_point/liouville_integration/six_point_build_momentum_banks.py) | Building coefficient tables and identifying repeated momentum nodes | CLI |
| [liouville_integration/six_point_collar_momentum_quadrature.py](code/tree_level/six_point/liouville_integration/six_point_collar_momentum_quadrature.py) | Momentum integration in $D_{\mathrm{sep}}$ and analytic radial integrals | Library |
| [liouville_integration/six_point_compare_momentum_quadratures.py](code/tree_level/six_point/liouville_integration/six_point_compare_momentum_quadratures.py) | Assessment of momentum integration and block truncation | CLI |
| [liouville_integration/six_point_deduplicate_momentum_nodes.py](code/tree_level/six_point/liouville_integration/six_point_deduplicate_momentum_nodes.py) | Building coefficient tables and identifying repeated momentum nodes | CLI |
| [liouville_integration/six_point_endpoint_and_tail_quadrature.py](code/tree_level/six_point/liouville_integration/six_point_endpoint_and_tail_quadrature.py) | Momentum integration grids | Library |
| [liouville_integration/six_point_mixed_order_node_catalog.py](code/tree_level/six_point/liouville_integration/six_point_mixed_order_node_catalog.py) | Building coefficient tables and identifying repeated momentum nodes | CLI |
| [liouville_integration/six_point_momentum_banks_and_reweighting.py](code/tree_level/six_point/liouville_integration/six_point_momentum_banks_and_reweighting.py) | Liouville momentum integration and stored coefficient tables | Library |
| [liouville_integration/six_point_momentum_control_proposals.py](code/tree_level/six_point/liouville_integration/six_point_momentum_control_proposals.py) | Momentum integration grids | Library |
| [liouville_integration/six_point_reuse_saved_coefficients.py](code/tree_level/six_point/liouville_integration/six_point_reuse_saved_coefficients.py) | Reusing previously computed coefficients | CLI |
| [moduli_integration/six_point_analytic_collar_subtraction.py](code/tree_level/six_point/moduli_integration/six_point_analytic_collar_subtraction.py) | Approximate integrand in $D_{\mathrm{sep}}$ and its integrated contribution | Library |
| [moduli_integration/six_point_assemble_curve_amplitudes.py](code/tree_level/six_point/moduli_integration/six_point_assemble_curve_amplitudes.py) | Combining Monte Carlo estimates and their covariance | CLI |
| [moduli_integration/six_point_assemble_radial_curve_covariance.py](code/tree_level/six_point/moduli_integration/six_point_assemble_radial_curve_covariance.py) | Combining Monte Carlo estimates and their covariance | CLI |
| [moduli_integration/six_point_batch_block_contraction.py](code/tree_level/six_point/moduli_integration/six_point_batch_block_contraction.py) | Weighted integrand $W_\Gamma$ | Library |
| [moduli_integration/six_point_boundary_strata.py](code/tree_level/six_point/moduli_integration/six_point_boundary_strata.py) | Separating domains and subtraction measure | Library |
| [moduli_integration/six_point_channel_coordinates.py](code/tree_level/six_point/moduli_integration/six_point_channel_coordinates.py) | Coordinate frames, transformations, and $\rho_\Gamma$ | Library |
| [moduli_integration/six_point_check_sampling_variance.py](code/tree_level/six_point/moduli_integration/six_point_check_sampling_variance.py) | Weight function and Monte Carlo sampling points | Library |
| [moduli_integration/six_point_collar_counterterm_geometry.py](code/tree_level/six_point/moduli_integration/six_point_collar_counterterm_geometry.py) | Separating domains and subtraction measure | Library |
| [moduli_integration/six_point_combine_truncation_orders.py](code/tree_level/six_point/moduli_integration/six_point_combine_truncation_orders.py) | Moduli integration with different block truncations | Library |
| [moduli_integration/six_point_estimate_collar_corrected_integrand.py](code/tree_level/six_point/moduli_integration/six_point_estimate_collar_corrected_integrand.py) | Separating subtraction and the remaining integrand | Library |
| [moduli_integration/six_point_evaluate_routed_geometry_shards.py](code/tree_level/six_point/moduli_integration/six_point_evaluate_routed_geometry_shards.py) | Separating subtraction and the remaining integrand | Library |
| [moduli_integration/six_point_integrate_collar_patch.py](code/tree_level/six_point/moduli_integration/six_point_integrate_collar_patch.py) | Approximate integrand in $D_{\mathrm{sep}}$ and its integrated contribution | CLI |
| [moduli_integration/six_point_integrate_mixed_order_amplitudes.py](code/tree_level/six_point/moduli_integration/six_point_integrate_mixed_order_amplitudes.py) | Moduli integration with different block truncations | CLI |
| [moduli_integration/six_point_prepare_radial_geometry.py](code/tree_level/six_point/moduli_integration/six_point_prepare_radial_geometry.py) | Weight function and Monte Carlo sampling points | CLI |
| [moduli_integration/six_point_radial_sampling_distribution.py](code/tree_level/six_point/moduli_integration/six_point_radial_sampling_distribution.py) | Weight function and Monte Carlo sampling points | Library |
| [moduli_integration/six_point_refinement_assembly.py](code/tree_level/six_point/moduli_integration/six_point_refinement_assembly.py) | Selected higher-order corrections and final amplitude assembly | CLI |
| [moduli_integration/six_point_route_collar_contributions.py](code/tree_level/six_point/moduli_integration/six_point_route_collar_contributions.py) | Separating subtraction and the remaining integrand | Library |
| [moduli_integration/six_point_sample_integration_geometry.py](code/tree_level/six_point/moduli_integration/six_point_sample_integration_geometry.py) | Coordinate frames, transformations, and $\rho_\Gamma$ | Library |
| [moduli_integration/six_point_stable_puncture_coordinates.py](code/tree_level/six_point/moduli_integration/six_point_stable_puncture_coordinates.py) | Coordinate frames, transformations, and $\rho_\Gamma$ | Library |
| [production/dispatch_shared_refinement_queue.py](code/tree_level/six_point/production/dispatch_shared_refinement_queue.py) | Checking saved inputs and distributing refinement jobs | CLI |
| [production/six_point_prepare_amplitude_curve.py](code/tree_level/six_point/production/six_point_prepare_amplitude_curve.py) | Preparing the amplitude calculation and running its stages | CLI |
| [production/six_point_refinement.py](code/tree_level/six_point/production/six_point_refinement.py) | Selective refinement through $N=14$ and continuation through $N=18$ | CLI |
| [production/six_point_run_frozen_energy_worker.py](code/tree_level/six_point/production/six_point_run_frozen_energy_worker.py) | Preparing the amplitude calculation and running its stages | CLI |
| [production/six_point_run_frozen_radial_worker.py](code/tree_level/six_point/production/six_point_run_frozen_radial_worker.py) | Preparing the amplitude calculation and running its stages | CLI |
| [production/six_point_submit_radial_campaign.py](code/tree_level/six_point/production/six_point_submit_radial_campaign.py) | Preparing the amplitude calculation and running its stages | CLI |
| [production/verify_saved_c14_continuation_inputs.py](code/tree_level/six_point/production/verify_saved_c14_continuation_inputs.py) | Checking saved inputs and distributing refinement jobs | Library |

</details>

<details>
<summary>Torus 1→1 — 13 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [conformal_block/torus_two_point_blocks.py](code/one_loop/two_point/conformal_block/torus_two_point_blocks.py) | Necklace and OPE conformal blocks, coefficient transformations, and direct descendant checks | Library |
| [liouville_integration/genus1_two_point_coefficient_bank.py](code/one_loop/two_point/liouville_integration/genus1_two_point_coefficient_bank.py) | Preparation and reuse of the two-point Liouville coefficient banks | CLI |
| [liouville_integration/genus1_two_point_worldsheet.py](code/one_loop/two_point/liouville_integration/genus1_two_point_worldsheet.py) | DOZZ factors, Liouville momentum quadrature, channel evaluation, and leading collision term | Library |
| [moduli_integration/genus1_two_point_thermal_integration.py](code/one_loop/two_point/moduli_integration/genus1_two_point_thermal_integration.py) | Thermal moduli integration over the full cusp and collision disc, with paired noncompact estimates | CLI |
| [moduli_integration/integrate_genus1_two_point_worldsheet.py](code/one_loop/two_point/moduli_integration/integrate_genus1_two_point_worldsheet.py) | Noncompact moduli integration and cusp checks | CLI |
| [other_scripts/checks/genus1_two_point_worldsheet_checks.py](code/one_loop/two_point/other_scripts/checks/genus1_two_point_worldsheet_checks.py) | Checks of the integrand, channel overlap, and modular covariance | CLI |
| [other_scripts/checks/torus_necklace_blocks_checks.py](code/one_loop/two_point/other_scripts/checks/torus_necklace_blocks_checks.py) | Independent checks of the necklace h-recursion | Script |
| [other_scripts/checks/torus_two_point_blocks_checks.py](code/one_loop/two_point/other_scripts/checks/torus_two_point_blocks_checks.py) | Independent block checks against direct descendants | Script |
| [other_scripts/fit_genus1_two_point_bry_scan.py](code/one_loop/two_point/other_scripts/fit_genus1_two_point_bry_scan.py) | Fits and convention checks for saved noncompact amplitudes | CLI |
| [other_scripts/genus1_two_point_thermal_production.py](code/one_loop/two_point/other_scripts/genus1_two_point_thermal_production.py) | Thermal radius scan using the saved production settings | CLI |
| [other_scripts/genus1_two_point_thermal_report.py](code/one_loop/two_point/other_scripts/genus1_two_point_thermal_report.py) | Assembly of the thermal amplitude, sampling errors, and radius figure | CLI |
| [other_scripts/plot_genus1_two_point_hybrid_recursion.py](code/one_loop/two_point/other_scripts/plot_genus1_two_point_hybrid_recursion.py) | Noncompact amplitude figure and saved recursion comparisons | CLI |
| [other_scripts/run_genus1_two_point_imaginary_scan.py](code/one_loop/two_point/other_scripts/run_genus1_two_point_imaginary_scan.py) | Noncompact amplitude scan used for comparison | CLI |

</details>

<details>
<summary>Torus 1→2, including thermal kernels — 61 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [conformal_block/genus1_collision_elliptic.py](code/one_loop/three_point/conformal_block/genus1_collision_elliptic.py) | Elliptic blocks for collision limits and precomputed coefficients | Library |
| [conformal_block/genus1_native_log_evaluation.py](code/one_loop/three_point/conformal_block/genus1_native_log_evaluation.py) | Evaluation of the retained native series without overflow | Library |
| [conformal_block/torus_c_recursion_numerics.py](code/one_loop/three_point/conformal_block/torus_c_recursion_numerics.py) | c-recursion and precision checks | Library |
| [conformal_block/torus_three_point_blocks.py](code/one_loop/three_point/conformal_block/torus_three_point_blocks.py) | Torus three-point blocks in the necklace, pair OPE, and comb channels | Library |
| [liouville_integration/genus1_boundary_banks.py](code/one_loop/three_point/liouville_integration/genus1_boundary_banks.py) | Saved native and elliptic coefficients and their truncations | Library |
| [liouville_integration/genus1_boundary_spectrum.py](code/one_loop/three_point/liouville_integration/genus1_boundary_spectrum.py) | Preparation of momentum nodes, DOZZ factors, and block coefficients | Library |
| [liouville_integration/genus1_channel_evaluation.py](code/one_loop/three_point/liouville_integration/genus1_channel_evaluation.py) | Channel evaluation and arithmetic checks | Library |
| [liouville_integration/genus1_collar_mp_arithmetic.py](code/one_loop/three_point/liouville_integration/genus1_collar_mp_arithmetic.py) | Channel evaluation and arithmetic checks | Library |
| [liouville_integration/genus1_native_bank.py](code/one_loop/three_point/liouville_integration/genus1_native_bank.py) | Saved native and elliptic coefficients and their truncations | Library |
| [liouville_integration/genus1_projection_momentum.py](code/one_loop/three_point/liouville_integration/genus1_projection_momentum.py) | Momentum rules for the degeneration terms | Library |
| [liouville_integration/genus1_projection_preparation.py](code/one_loop/three_point/liouville_integration/genus1_projection_preparation.py) | Preparation of momentum nodes, DOZZ factors, and block coefficients | CLI |
| [liouville_integration/genus1_three_point_momentum.py](code/one_loop/three_point/liouville_integration/genus1_three_point_momentum.py) | Momentum rules and convergence checks | Library |
| [liouville_integration/genus1_three_point_worldsheet.py](code/one_loop/three_point/liouville_integration/genus1_three_point_worldsheet.py) | Liouville correlator and momentum integrand | Library |
| [liouville_integration/thermal/banks.py](code/one_loop/three_point/liouville_integration/thermal/banks.py) | Thermal coefficient banks and saved momentum integrals | Library |
| [liouville_integration/thermal/evaluation.py](code/one_loop/three_point/liouville_integration/thermal/evaluation.py) | Evaluation of the thermal momentum integrand | Library |
| [liouville_integration/thermal/paired_quadrature.py](code/one_loop/three_point/liouville_integration/thermal/paired_quadrature.py) | Thermal momentum rules and node placement | Script |
| [liouville_integration/thermal/quadrature.py](code/one_loop/three_point/liouville_integration/thermal/quadrature.py) | Thermal momentum rules and node placement | Library |
| [liouville_integration/thermal/thermal_cut_rule.py](code/one_loop/three_point/liouville_integration/thermal/thermal_cut_rule.py) | Thermal momentum rules and node placement | Script |
| [liouville_integration/thermal/thermal_saved_inputs.py](code/one_loop/three_point/liouville_integration/thermal/thermal_saved_inputs.py) | Thermal coefficient banks and saved momentum integrals | Library |
| [moduli_integration/genus1_analytic_boundary_layer.py](code/one_loop/three_point/moduli_integration/genus1_analytic_boundary_layer.py) | Partition of moduli space, sampling weights, and collision-region maps | Library |
| [moduli_integration/genus1_boundary_batch.py](code/one_loop/three_point/moduli_integration/genus1_boundary_batch.py) | Reuse of moduli-dependent factors | Library |
| [moduli_integration/genus1_boundary_cubature.py](code/one_loop/three_point/moduli_integration/genus1_boundary_cubature.py) | Deterministic quadrature and assembly of the moduli integral | Library |
| [moduli_integration/genus1_boundary_integration.py](code/one_loop/three_point/moduli_integration/genus1_boundary_integration.py) | Deterministic quadrature and assembly of the moduli integral | Library |
| [moduli_integration/genus1_collision_sampling.py](code/one_loop/three_point/moduli_integration/genus1_collision_sampling.py) | Partition of moduli space, sampling weights, and collision-region maps | Library |
| [moduli_integration/genus1_comb_collar.py](code/one_loop/three_point/moduli_integration/genus1_comb_collar.py) | Collision subtractions and integrated degeneration terms | Library |
| [moduli_integration/genus1_cusp_control.py](code/one_loop/three_point/moduli_integration/genus1_cusp_control.py) | Nonseparating approximation and tube integrals | Library |
| [moduli_integration/genus1_integrated_scale_collar.py](code/one_loop/three_point/moduli_integration/genus1_integrated_scale_collar.py) | Collision subtractions and integrated degeneration terms | Library |
| [moduli_integration/genus1_modular_geometry.py](code/one_loop/three_point/moduli_integration/genus1_modular_geometry.py) | Plumbing parameters, channel choice, and sewing domains | Library |
| [moduli_integration/genus1_moduli_panels.py](code/one_loop/three_point/moduli_integration/genus1_moduli_panels.py) | Deterministic quadrature and assembly of the moduli integral | Library |
| [moduli_integration/genus1_moduli_sampling.py](code/one_loop/three_point/moduli_integration/genus1_moduli_sampling.py) | Partition of moduli space, sampling weights, and collision-region maps | Library |
| [moduli_integration/genus1_nonseparating.py](code/one_loop/three_point/moduli_integration/genus1_nonseparating.py) | Nonseparating approximation and tube integrals | Library |
| [moduli_integration/genus1_pair_collar.py](code/one_loop/three_point/moduli_integration/genus1_pair_collar.py) | Collision subtractions and integrated degeneration terms | Library |
| [moduli_integration/genus1_sewing_domains.py](code/one_loop/three_point/moduli_integration/genus1_sewing_domains.py) | Plumbing parameters, channel choice, and sewing domains | Library |
| [moduli_integration/genus1_three_point_channel_atlas.py](code/one_loop/three_point/moduli_integration/genus1_three_point_channel_atlas.py) | Plumbing parameters, channel choice, and sewing domains | Library |
| [moduli_integration/genus1_three_point_collision.py](code/one_loop/three_point/moduli_integration/genus1_three_point_collision.py) | Collision subtractions and integrated degeneration terms | Library |
| [moduli_integration/thermal/kernels.py](code/one_loop/three_point/moduli_integration/thermal/kernels.py) | Thermal kinematics and integrated boundary kernels | Script |
| [moduli_integration/thermal/radius_scan.py](code/one_loop/three_point/moduli_integration/thermal/radius_scan.py) | Moduli sampling for the thermal correction | CLI |
| [moduli_integration/thermal/remainders.py](code/one_loop/three_point/moduli_integration/thermal/remainders.py) | Thermal remainder terms | Library |
| [moduli_integration/thermal/study_pair_sampling.py](code/one_loop/three_point/moduli_integration/thermal/study_pair_sampling.py) | Moduli sampling for the thermal correction | CLI |
| [other_scripts/checks/genus1_three_point_c_recursion_checks.py](code/one_loop/three_point/other_scripts/checks/genus1_three_point_c_recursion_checks.py) | Independent checks of the block recursion | CLI |
| [other_scripts/genus1_amplitude_curve.py](code/one_loop/three_point/other_scripts/genus1_amplitude_curve.py) | Production inputs and preparation of the energy scan | CLI |
| [other_scripts/genus1_amplitude_report.py](code/one_loop/three_point/other_scripts/genus1_amplitude_report.py) | Amplitude reports and numerical assessment | CLI |
| [other_scripts/genus1_compact_replay.py](code/one_loop/three_point/other_scripts/genus1_compact_replay.py) | Production of the compact and degeneration contributions | CLI |
| [other_scripts/genus1_final_curve_plan.py](code/one_loop/three_point/other_scripts/genus1_final_curve_plan.py) | Production inputs and preparation of the energy scan | CLI |
| [other_scripts/genus1_fresh_analytic.py](code/one_loop/three_point/other_scripts/genus1_fresh_analytic.py) | Production of the compact and degeneration contributions | CLI |
| [other_scripts/genus1_moduli_amplitude.py](code/one_loop/three_point/other_scripts/genus1_moduli_amplitude.py) | Production of the compact and degeneration contributions | CLI |
| [other_scripts/genus1_moduli_panel_campaign.py](code/one_loop/three_point/other_scripts/genus1_moduli_panel_campaign.py) | Production of the compact and degeneration contributions | CLI |
| [other_scripts/genus1_moduli_stability.py](code/one_loop/three_point/other_scripts/genus1_moduli_stability.py) | Moduli-quadrature comparisons | CLI |
| [other_scripts/genus1_moduli_stability_curve.py](code/one_loop/three_point/other_scripts/genus1_moduli_stability_curve.py) | Moduli-quadrature comparisons | CLI |
| [other_scripts/genus1_pair_cell_refinement.py](code/one_loop/three_point/other_scripts/genus1_pair_cell_refinement.py) | Refinement of selected integration regions | CLI |
| [other_scripts/genus1_primary_first_production.py](code/one_loop/three_point/other_scripts/genus1_primary_first_production.py) | Production inputs and preparation of the energy scan | CLI |
| [other_scripts/genus1_primary_moduli_refinement.py](code/one_loop/three_point/other_scripts/genus1_primary_moduli_refinement.py) | Refinement of selected integration regions | CLI |
| [other_scripts/genus1_production_support.py](code/one_loop/three_point/other_scripts/genus1_production_support.py) | Production settings, checked records, and source paths | Library |
| [other_scripts/genus1_rebuild_policy.py](code/one_loop/three_point/other_scripts/genus1_rebuild_policy.py) | Amplitude reports and numerical assessment | Script |
| [other_scripts/genus1_source_paths.py](code/one_loop/three_point/other_scripts/genus1_source_paths.py) | Production settings, checked records, and source paths | Library |
| [other_scripts/genus1_upper_pair_refinement.py](code/one_loop/three_point/other_scripts/genus1_upper_pair_refinement.py) | Refinement of selected integration regions | CLI |
| [other_scripts/plot_genus1_production.py](code/one_loop/three_point/other_scripts/plot_genus1_production.py) | Amplitude figure and numerical diagnostics | Script |
| [other_scripts/thermal/draft_radius_campaign.py](code/one_loop/three_point/other_scripts/thermal/draft_radius_campaign.py) | Thermal production, assembly, and momentum comparisons | CLI |
| [other_scripts/thermal/draft_radius_collect.py](code/one_loop/three_point/other_scripts/thermal/draft_radius_collect.py) | Thermal production, assembly, and momentum comparisons | CLI |
| [other_scripts/thermal/draft_radius_validation.py](code/one_loop/three_point/other_scripts/thermal/draft_radius_validation.py) | Thermal production, assembly, and momentum comparisons | CLI |
| [other_scripts/thermal/reference_campaign.py](code/one_loop/three_point/other_scripts/thermal/reference_campaign.py) | Thermal production, assembly, and momentum comparisons | CLI |

</details>

<details>
<summary>Genus-two free energy and channel matching — 45 files</summary>

| File | Purpose | Interface |
| --- | --- | --- |
| [conformal_block/ccy_genus2_glasses_block.py](code/two_loop/conformal_block/ccy_genus2_glasses_block.py) | Genus-two conformal block in the glasses channel | CLI |
| [geometry/audit_q_to_omega_accuracy.py](code/two_loop/geometry/audit_q_to_omega_accuracy.py) | Checks of the period map and modular transformations | CLI |
| [geometry/bolza_ccy_recursion.py](code/two_loop/geometry/bolza_ccy_recursion.py) | Checks of the period map and modular transformations | CLI |
| [geometry/bolza_torus_plumbing_reach.py](code/two_loop/geometry/bolza_torus_plumbing_reach.py) | Plumbing atlas and choice of the homology marking | CLI |
| [geometry/genus2_calibrated_schottky.py](code/two_loop/geometry/genus2_calibrated_schottky.py) | Period map and choice of its numerical implementation | Library |
| [geometry/genus2_holomorphic_period_table.py](code/two_loop/geometry/genus2_holomorphic_period_table.py) | Period-map tables and their construction | Library |
| [geometry/genus2_hybrid_period_map.py](code/two_loop/geometry/genus2_hybrid_period_map.py) | Period map and choice of its numerical implementation | Library |
| [geometry/genus2_integration_geometry.py](code/two_loop/geometry/genus2_integration_geometry.py) | Channel selection and reconstruction of saved plumbing coordinates | Library |
| [geometry/genus2_multiprecision_collocation.py](code/two_loop/geometry/genus2_multiprecision_collocation.py) | Period map and choice of its numerical implementation | Library |
| [geometry/genus2_period_table.py](code/two_loop/geometry/genus2_period_table.py) | Period-map tables and their construction | Library |
| [geometry/genus2_period_table_grid.py](code/two_loop/geometry/genus2_period_table_grid.py) | Period-map tables and their construction | CLI |
| [geometry/genus2_period_table_selector.py](code/two_loop/geometry/genus2_period_table_selector.py) | Channel selection and reconstruction of saved plumbing coordinates | Library |
| [geometry/genus2_plumbing_atlas.py](code/two_loop/geometry/genus2_plumbing_atlas.py) | Plumbing atlas and choice of the homology marking | CLI |
| [geometry/liouville_genus2_modular_check.py](code/two_loop/geometry/liouville_genus2_modular_check.py) | Checks of the period map and modular transformations | CLI |
| [liouville_integration/genus2_liouville_evaluation.py](code/two_loop/liouville_integration/genus2_liouville_evaluation.py) | Channel evaluation and logarithmic scaling of the Liouville partition function | Library |
| [liouville_integration/liouville_genus2.py](code/two_loop/liouville_integration/liouville_genus2.py) | Separating approximation in the pair-of-tori channel | CLI |
| [liouville_integration/liouville_genus2_ccy.py](code/two_loop/liouville_integration/liouville_genus2_ccy.py) | Liouville momentum integrals in the theta and glasses channels | CLI |
| [liouville_integration/liouville_genus2_glasses.py](code/two_loop/liouville_integration/liouville_genus2_glasses.py) | Liouville momentum integrals in the theta and glasses channels | CLI |
| [moduli_integration/genus2_bc_ghost.py](code/two_loop/moduli_integration/genus2_bc_ghost.py) | Ghost and matter factors and the complete moduli integrand | CLI |
| [moduli_integration/genus2_c1_string_integrand.py](code/two_loop/moduli_integration/genus2_c1_string_integrand.py) | Ghost and matter factors and the complete moduli integrand | Library |
| [moduli_integration/genus2_moduli_estimation.py](code/two_loop/moduli_integration/genus2_moduli_estimation.py) | Sampling weights, moduli estimates, and their statistical errors | Library |
| [moduli_integration/genus2_moduli_sampling.py](code/two_loop/moduli_integration/genus2_moduli_sampling.py) | Siegel--Gottschling fundamental domain and Monte Carlo sampling | CLI |
| [moduli_integration/genus2_node_evaluation.py](code/two_loop/moduli_integration/genus2_node_evaluation.py) | Ghost and matter factors and the complete moduli integrand | Library |
| [moduli_integration/genus2_scalar_evaluation.py](code/two_loop/moduli_integration/genus2_scalar_evaluation.py) | Scalar partition function and its truncation checks | Library |
| [moduli_integration/genus2_separating_tail.py](code/two_loop/moduli_integration/genus2_separating_tail.py) | Separating approximation, radial integral, and sampled remainder | Library |
| [moduli_integration/genus2_siegel_fundamental_domain.py](code/two_loop/moduli_integration/genus2_siegel_fundamental_domain.py) | Siegel--Gottschling fundamental domain and Monte Carlo sampling | CLI |
| [other_scripts/assemble_genus2_resolved_production.py](code/two_loop/other_scripts/assemble_genus2_resolved_production.py) | Combining bulk and separating contributions | CLI |
| [other_scripts/assemble_genus2_tail_completion.py](code/two_loop/other_scripts/assemble_genus2_tail_completion.py) | Combining bulk and separating contributions | CLI |
| [other_scripts/audit_genus2_g_convention.py](code/two_loop/other_scripts/audit_genus2_g_convention.py) | Radius dependence, figures, and checks of normalization conventions | Script |
| [other_scripts/check_channel_convergence.py](code/two_loop/other_scripts/check_channel_convergence.py) | Compare integrated Liouville values at fixed table geometry while changing momentum and block orders. | CLI |
| [other_scripts/checks/ccy_genus2_glasses_block_checks.py](code/two_loop/other_scripts/checks/ccy_genus2_glasses_block_checks.py) | Independent checks of the glasses conformal block | Script |
| [other_scripts/checks/genus2_c1_string_integrand_checks.py](code/two_loop/other_scripts/checks/genus2_c1_string_integrand_checks.py) | Checks of the moduli integrand | Script |
| [other_scripts/checks/genus2_separating_tail_checks.py](code/two_loop/other_scripts/checks/genus2_separating_tail_checks.py) | Checks of the separating approximation, precision, and tail assembly | CLI |
| [other_scripts/checks/genus2_tail_completion_checks.py](code/two_loop/other_scripts/checks/genus2_tail_completion_checks.py) | Checks of the separating approximation, precision, and tail assembly | Script |
| [other_scripts/checks/genus2_tail_precision_checks.py](code/two_loop/other_scripts/checks/genus2_tail_precision_checks.py) | Checks of the separating approximation, precision, and tail assembly | Script |
| [other_scripts/checks/plot_genus2_absolute_bry_checks.py](code/two_loop/other_scripts/checks/plot_genus2_absolute_bry_checks.py) | Checks of the plotted free energy | Script |
| [other_scripts/genus2_glasses_production_gate.py](code/two_loop/other_scripts/genus2_glasses_production_gate.py) | Bulk production and checks of numerical convergence | CLI |
| [other_scripts/genus2_moduli_production.py](code/two_loop/other_scripts/genus2_moduli_production.py) | Bulk production and checks of numerical convergence | CLI |
| [other_scripts/genus2_separating_production.py](code/two_loop/other_scripts/genus2_separating_production.py) | Separating-region production and refinement of saved nodes | CLI |
| [other_scripts/genus2_separating_residual_retry.py](code/two_loop/other_scripts/genus2_separating_residual_retry.py) | Separating-region production and refinement of saved nodes | CLI |
| [other_scripts/plot_genus2_absolute_bry.py](code/two_loop/other_scripts/plot_genus2_absolute_bry.py) | Radius dependence, figures, and checks of normalization conventions | CLI |
| [other_scripts/rebuild_genus2_bulk_control_groups.py](code/two_loop/other_scripts/rebuild_genus2_bulk_control_groups.py) | Reconstructing bulk control groups and the separating correction | CLI |
| [other_scripts/recompute_channel_comparison.py](code/two_loop/other_scripts/recompute_channel_comparison.py) | Recompute theta/glasses Q_L at the five printed manuscript period matrices. | CLI |
| [other_scripts/recover_channel_convergence.py](code/two_loop/other_scripts/recover_channel_convergence.py) | Verify fixed-geometry convergence receipts and recover observed relative changes. | CLI |
| [other_scripts/study_genus2_separating_tail_correction.py](code/two_loop/other_scripts/study_genus2_separating_tail_correction.py) | Reconstructing bulk control groups and the separating correction | CLI |

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
