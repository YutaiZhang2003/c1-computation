#!/usr/bin/env python3
"""Assemble audited per-energy receipts for the sphere 1->5 curve."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[4]/'code/general/conventions'))
from c1_observables import write_six_point_amplitudes


def atomic_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main(root):
    root = Path(root).resolve()
    campaign = json.loads((root / "campaign.json").read_text())
    rows, empirical_failures = [], []
    for energy in campaign["energies"]:
        label = energy["label"]
        result = json.loads((root / label / "final_audit.json").read_text())
        if (float(result.get("t", -1)) != float(energy["t"])
                or result.get("sample_count") != 2048
                or not result.get("finite")
                or not result.get("exact_complete_source")
                or result.get("unresolved_scope_count") != 0
                or result.get("numerical_failure_count") != 0):
            raise RuntimeError("energy audit is incomplete or inconsistent: " + label)
        if not result.get("one_percent_empirical_screen"):
            empirical_failures.append(label)
        rows.append(result)
    if len(rows) != len(campaign["energies"]):
        raise RuntimeError("curve aggregate omitted an energy")
    payload = dict(version="sphere-six-point-yin-curve-results-v3", results=rows,
        complete_execution=True, all_energies_finite=True,
        one_percent_empirical_screen_all=not empirical_failures,
        one_percent_empirical_screen_failures=empirical_failures,
        total_accuracy_certified=False)
    output = root / "curve_results.json"
    if output.exists() and json.loads(output.read_text()) != payload:
        raise ValueError("existing curve aggregate differs from current audited inputs")
    if not output.exists():
        atomic_write(output, payload)
    write_six_point_amplitudes(root, payload)
    print(json.dumps(dict(energies=len(rows), empirical_failures=empirical_failures,
                          output=str(output)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    main(args.campaign_root)
