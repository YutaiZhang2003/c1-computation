#!/usr/bin/env python3
"""Run a packaged module, optionally in a separate writable data workspace."""
from pathlib import Path
import argparse
import json
import importlib.util
import os
import runpy
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def bootstrap(include_archived=False):
    catalog = json.loads((ROOT / "PACKAGE_MAP.json").read_text())
    # New research modules are registered without rewriting frozen provenance.
    for name, path in (
        ("export_current_convention_results", "code/figures/export_current_convention_results.py"),
        ("export_genus1_collision_plot_data", "code/one_loop/analysis/export_genus1_collision_plot_data.py"),
        ("plot_genus1_collision_final", "code/one_loop/analysis/plot_genus1_collision_final.py"),
        ("export_amplitude_component_plot_data", "code/figures/export_amplitude_component_plot_data.py"),
        ("c1_observables", "code/general/conventions/c1_observables.py"),
        ("c1_observable_convention_checks", "code/general/conventions/c1_observable_convention_checks.py"),
        ("genus1_adaptive_rerun", "code/one_loop/analysis/genus1_adaptive_rerun.py"),
        ("genus1_adaptive_cft_checks", "code/one_loop/validation/genus1_adaptive_cft_checks.py"),
        ("genus1_modular_run", "code/one_loop/production/genus1_modular_run.py"),
        ("genus1_modular_cft_checks", "code/one_loop/validation/genus1_modular_cft_checks.py"),
        ("genus1_plumbing_assignment_checks", "code/one_loop/validation/genus1_plumbing_assignment_checks.py"),
        ("genus1_plumbing_survey", "code/one_loop/analysis/genus1_plumbing_survey.py"),
        ("sphere_six_point_coefficient_provider_checks", "code/tree_level/validation/sphere_six_point_coefficient_provider_checks.py"),
        ("sphere_six_point_high_order_pbw_checks", "code/tree_level/validation/sphere_six_point_high_order_pbw_checks.py"),
        ("sphere_six_point_archive_replay_checks", "code/tree_level/validation/sphere_six_point_archive_replay_checks.py"),
        ("sphere_six_point_optimized_geometry_checks", "code/tree_level/validation/sphere_six_point_optimized_geometry_checks.py"),
        ("sphere_six_point_radial_proposal_checks", "code/tree_level/validation/sphere_six_point_radial_proposal_checks.py"),
        ("sphere_six_point_native_geometry_checks", "code/tree_level/validation/sphere_six_point_native_geometry_checks.py"),
        ("sphere_six_point_channel_routing_checks", "code/tree_level/validation/sphere_six_point_channel_routing_checks.py"),
        ("sphere_six_point_routing_integration_checks", "code/tree_level/validation/sphere_six_point_routing_integration_checks.py"),
        ("sphere_six_point_momentum_banks_checks", "code/tree_level/validation/sphere_six_point_momentum_banks_checks.py"),
        ("sphere_six_point_bank_build_checks", "code/tree_level/validation/sphere_six_point_bank_build_checks.py"),
        ("sphere_six_point_radial_forest_checks", "code/tree_level/validation/sphere_six_point_radial_forest_checks.py"),
        ("sphere_six_point_optimized_scan_checks", "code/tree_level/validation/sphere_six_point_optimized_scan_checks.py"),
        ("sphere_six_point_bank_build", "code/tree_level/production/sphere_six_point_bank_build.py"),
        ("sphere_six_point_archive_replay", "code/tree_level/production/sphere_six_point_archive_replay.py"),
        ("sphere_six_point_radial_proposal", "code/tree_level/kernels/sphere_six_point_radial_proposal.py"),
        ("sphere_six_point_prepare_radial_mixture_geometry", "code/tree_level/production/sphere_six_point_prepare_radial_mixture_geometry.py"),
        ("sphere_six_point_optimized_scan", "code/tree_level/production/sphere_six_point_optimized_scan.py"),
        ("sphere_six_point_prepare_routing", "code/tree_level/production/sphere_six_point_prepare_routing.py"),
        ("analyze_sphere_six_point_channel_routing", "code/tree_level/analysis/analyze_sphere_six_point_channel_routing.py"),
        ("genus1_three_point_c_recursion_checks", "code/one_loop/validation/genus1_three_point_c_recursion_checks.py"),
        ("genus1_three_point_collision_checks", "code/one_loop/validation/genus1_three_point_collision_checks.py"),
        ("genus1_native_sewing_checks", "code/one_loop/validation/genus1_native_sewing_checks.py"),
        ("sphere_six_point_hybrid_scan", "code/tree_level/production/sphere_six_point_hybrid_scan.py"),
        ("sphere_six_point_elliptic_checks", "code/tree_level/validation/sphere_six_point_elliptic_checks.py"),
        ("sphere_six_point_hybrid_summary", "code/tree_level/analysis/sphere_six_point_hybrid_summary.py"),
        (
            "genus1_three_point_nonseparating_tail",
            "code/one_loop/kernels/genus1_three_point_nonseparating_tail.py",
        ),
        (
            "genus1_three_point_nonseparating_tail_checks",
            "code/one_loop/validation/genus1_three_point_nonseparating_tail_checks.py",
        ),
        (
            "integrate_genus1_three_point_nonseparating_tail",
            "code/one_loop/production/integrate_genus1_three_point_nonseparating_tail.py",
        ),
        (
            "genus1_three_point_nonseparating_tail_pilot",
            "code/one_loop/production/genus1_three_point_nonseparating_tail_pilot.py",
        ),
        (
            "assemble_genus1_three_point_nonseparating_tail",
            "code/one_loop/analysis/assemble_genus1_three_point_nonseparating_tail.py",
        ),
    ):
        if (ROOT / path).exists():
            catalog["module_map"][name] = path
    layout_path = ROOT / "code/tree_level/LAYOUT.json"
    if layout_path.is_file():
        layout = json.loads(layout_path.read_text())
        catalog["module_map"] = {
            name: path for name, path in catalog["module_map"].items()
            if not path.startswith("code/tree_level/")
        }
        catalog["module_aliases"] = layout.get("module_aliases", {})
        catalog["entrypoint_arguments"] = layout.get("entrypoint_arguments", {})
        archived = set(layout["archived_modules"])
        catalog["module_map"].update({
            name: path for name, path in layout["module_map"].items()
            if include_archived or name not in archived
        })
    one_loop_layout_path = ROOT / "code/one_loop/LAYOUT.json"
    if one_loop_layout_path.is_file():
        one_loop_layout = json.loads(one_loop_layout_path.read_text())
        catalog["module_map"] = {
            name: path for name, path in catalog["module_map"].items()
            if not path.startswith("code/one_loop/")
        }
        archived = set(one_loop_layout["archived_modules"])
        catalog["module_map"].update({
            name: path for name, path in one_loop_layout["module_map"].items()
            if include_archived or name not in archived
        })
    paths = {str((ROOT / p).parent) for p in catalog["module_map"].values()}
    paths.add(str(ROOT / "code/general/io"))
    # Remove inherited local import roots. Standard library and installed
    # third-party packages remain; none of the old checkout is an import root.
    retained = [p for p in sys.path if p and Path(p).resolve() != Path.cwd().resolve()
                and not (Path(p).resolve() == ROOT.parent.parent.resolve())]
    sys.path[:] = sorted(paths) + retained
    sys.dont_write_bytecode = True
    if layout_path.is_file() and catalog.get("module_aliases"):
        helper_name = "tree_level_source_compatibility"
        helper_path = ROOT / layout["module_map"][helper_name]
        helper = sys.modules.get(helper_name)
        if helper is None or Path(helper.__file__).resolve() != helper_path.resolve():
            spec = importlib.util.spec_from_file_location(helper_name, helper_path)
            helper = importlib.util.module_from_spec(spec)
            sys.modules[helper_name] = helper
            spec.loader.exec_module(helper)
        helper.install_module_aliases(catalog["module_aliases"], layout.get("symbol_aliases"))
    if one_loop_layout_path.is_file():
        import genus1_source_paths
        genus1_source_paths.install()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "c1-package-matplotlib"))
    return catalog


def protect_snapshot(event, args):
    def protected(value):
        path = Path(os.fsdecode(value)).resolve()
        return path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "output")

    if event in {"os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.symlink",
                 "os.link", "os.chmod", "os.truncate"}:
        count = 2 if event in {"os.rename", "os.symlink", "os.link"} else 1
        for value in args[:count]:
            if isinstance(value, (str, bytes, os.PathLike)):
                if protected(value):
                    raise PermissionError("Source and data are immutable; write under output/ or use --work-dir")
    if event == "open":
        filename, mode, flags = args
        if not isinstance(filename, (str, bytes, os.PathLike)):
            return
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing:
            if protected(filename):
                raise PermissionError("Source and data are immutable; write under output/ or use --work-dir")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, help="Copy frozen data once to this new workspace; write results there.")
    parser.add_argument("--list", action="store_true", help="List packaged modules and their purposes.")
    parser.add_argument("--include-archived", action="store_true",
                        help="Also expose archived modules; historical runs require their frozen sources.")
    parser.add_argument("module", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    catalog = bootstrap(include_archived=args.include_archived)
    if args.list:
        for name, path in sorted(catalog["module_map"].items()):
            print(f"{name:52} {path}")
        return
    if not args.module:
        parser.error("choose a module, or --list")
    name = Path(args.module).stem
    args.arguments = catalog.get("entrypoint_arguments", {}).get(name, []) + args.arguments
    name = catalog.get("module_aliases", {}).get(name, name)
    if name not in catalog["module_map"]:
        parser.error("module is not part of this package: " + name)
    work = ROOT
    if args.work_dir is not None:
        work = args.work_dir.expanduser().resolve()
        if work.is_relative_to(ROOT) or ROOT.is_relative_to(work):
            parser.error("--work-dir must be separate from the package and its ancestors")
        work.mkdir(parents=True, exist_ok=True)
        marker = work / ".c1-workspace.json"
        if not marker.exists():
            if (work / "data").exists():
                parser.error("unrecognized data directory in the requested workspace")
            shutil.copytree(ROOT / "data", work / "data")
            marker.write_text(json.dumps({"package": catalog["package"]}, indent=2) + "\n")
        elif json.loads(marker.read_text()).get("package") != catalog["package"]:
            parser.error("workspace belongs to a different package")
    os.environ["C1_WORKSPACE"] = str(work)
    os.chdir(work)
    sys.argv = [str(ROOT / catalog["module_map"][name])] + args.arguments
    sys.addaudithook(protect_snapshot)
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
