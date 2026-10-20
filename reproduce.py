#!/usr/bin/env python3
"""Verify the package and independently recover the draft's numerical results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def verify_manifest(root, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            raise ValueError("Missing package file: " + item["path"])
        contents = path.read_bytes()
        if len(contents) != item["bytes"] or hashlib.sha256(contents).hexdigest() != item["sha256"]:
            raise ValueError("Changed package file: " + item["path"])
    return len(manifest["files"])


def compare_outputs(actual, reference):
    """Compare every saved numerical table and every covariance-array field."""
    import numpy as np
    comparisons = []
    for source in sorted(reference.rglob("*")):
        if source.suffix not in {".csv", ".dat", ".npz"}:
            continue
        relative = source.relative_to(reference)
        target = actual / relative
        if source.suffix == ".npz":
            with np.load(source, allow_pickle=False) as a, np.load(target, allow_pickle=False) as b:
                if set(a.files) != set(b.files):
                    raise ValueError("Different array fields: " + str(relative))
                for key in a.files:
                    np.testing.assert_allclose(b[key], a[key], rtol=2e-10, atol=1e-28,
                                               err_msg=f"{relative}: {key}")
        else:
            delimiter = "," if source.suffix == ".csv" else " "
            with source.open(newline="") as left, target.open(newline="") as right:
                saved = list(csv.reader(left, delimiter=delimiter))
                recovered = list(csv.reader(right, delimiter=delimiter))
            if len(saved) != len(recovered) or saved[0] != recovered[0]:
                raise ValueError("Different table dimensions/header: " + str(relative))
            for a, b in zip(saved[1:], recovered[1:]):
                if len(a) != len(b):
                    raise ValueError("Different row widths: " + str(relative))
                for x, y in zip(a, b):
                    try:
                        numbers = float(y), float(x)
                    except ValueError:
                        if x != y:
                            raise ValueError("Different table labels: " + str(relative))
                    else:
                        np.testing.assert_allclose(*numbers, rtol=2e-10, atol=1e-28,
                                                   err_msg=str(relative))
        comparisons.append(str(relative))
    if len(comparisons) != 21:
        raise ValueError(f"Expected 21 reference tables/arrays; found {len(comparisons)}")
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"output/recovered",
                        help="New directory for regenerated tables and plots")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check source, input and reference checksums without recovery")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip PDF/PNG rendering")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or later is required")
    if sys.flags.optimize:
        parser.error("Run without -O: verification requires assertions")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import run
    run.bootstrap()
    from recover_production_results import verify_data, main as recover
    source_count = verify_manifest(ROOT, ROOT/"SOURCE_MANIFEST.json")
    reference = ROOT/"data/reference"
    reference_count = verify_manifest(reference, reference/"RESULTS_MANIFEST.json")
    input_report = verify_data(ROOT/"data/inputs")
    channel_data = ROOT/"data/channel_comparison"
    channel_count = verify_manifest(channel_data, channel_data/"MANIFEST.json")
    print(f"Verified {source_count} source files, {input_report['files']} inputs, "
          f"{reference_count} reference files, and {channel_count} channel-comparison files.",
          flush=True)
    if args.verify_only:
        return
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("The output directory already exists; choose a new --output")
    protected = [ROOT/"code", ROOT/"data", ROOT/".git"]
    if output == ROOT or any(output.is_relative_to(path) for path in protected):
        parser.error("Choose an output outside code/, data/, and .git/")
    arguments = ["recover_production_results.py", "--output", str(output)]
    if not args.no_figures:
        arguments.append("--figures")
    sys.argv = arguments
    recover()
    comparisons = compare_outputs(output, reference)
    from recompute_channel_comparison import recover_receipts
    channel_report = recover_receipts(
        channel_data/"receipts", output/"two_loop/channel_comparison")
    report_path = output/"verification.json"
    report = json.loads(report_path.read_text())
    report["standalone_reference_comparison"] = {
        "passed": True, "files": comparisons, "rtol": 2e-10, "atol": 1e-28,
        "scope": "All retained CSV/DAT tables and all fields of every retained covariance NPZ",
    }
    report["channel_comparison"] = channel_report
    import importlib.metadata
    report["environment"] = {
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "mpmath", "sympy", "matplotlib", "pillow")},
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"PASS: six result groups; {len(comparisons)} reference tables/arrays; "
          f"five channel-comparison rows; "
          f"{8 if not args.no_figures else 0} numerical panels.\nOutput: {output}")


if __name__ == "__main__":
    main()
