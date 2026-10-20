"""Package-local paths, including relocation of immutable source manifests.

No original checkout or cluster directory is consulted. Numerical artifacts
retain their original bytes; an explicit manifest maps their provenance paths
to the copies in this package (or an isolated reproduction workspace).
"""
from pathlib import Path
import json
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_MAP = json.loads((PACKAGE_ROOT / "PACKAGE_MAP.json").read_text())
_TREE_LAYOUT = PACKAGE_ROOT / "code/tree_level/LAYOUT.json"
_TREE_PATHS = {}
_TREE_ALIASES = {}
if _TREE_LAYOUT.is_file():
    _tree = json.loads(_TREE_LAYOUT.read_text())
    _MAP["module_map"].update(_tree["module_map"])
    _TREE_PATHS = _tree["path_map"]
    _TREE_ALIASES = _tree.get("module_aliases", {})
_ONE_LOOP_LAYOUT = PACKAGE_ROOT / "code/one_loop/LAYOUT.json"
if _ONE_LOOP_LAYOUT.is_file():
    _one_loop = json.loads(_ONE_LOOP_LAYOUT.read_text())
    _MAP["module_map"].update(_one_loop["module_map"])
    _TREE_PATHS.update(_one_loop["path_map"])


def workspace_root():
    return Path(os.environ.get("C1_WORKSPACE", str(PACKAGE_ROOT))).resolve()


def data_root(scope):
    if scope not in {"tree_level", "one_loop", "two_loop"}:
        raise ValueError("Unknown computation scope: " + scope)
    return workspace_root() / "data" / "production" / scope


def module_path(name):
    name = Path(name).stem
    return PACKAGE_ROOT / _MAP["module_map"][_TREE_ALIASES.get(name, name)]


def package_path(value):
    """Resolve known source paths; preserve explicitly supplied new paths."""
    text = str(value).replace("\\", "/")
    target = _TREE_PATHS.get(text, _MAP["remote_path_aliases"].get(text))
    if target is None:
        origin = text
        for marker in ("plumbing/results/", "output/data/", "plumbing/config/"):
            if marker in text:
                origin = marker + text.split(marker, 1)[1]
                break
        target = _MAP["path_map"].get(origin)
        if target is None and origin.startswith("plumbing/results/"):
            tail = origin[len("plumbing/results/"):]
            first = tail.split("/")[0]
            scope = ("tree_level" if first.startswith(("sphere_", "genus0_")) else
                     "one_loop" if first.startswith("genus1_") else "two_loop")
            target = "data set/" + scope + "/" + tail
        elif target is None and origin.startswith("output/data/"):
            target = "data set/two_loop/releases/" + origin[len("output/data/"):]
    if target is not None:
        target = _TREE_PATHS.get(target, target)
        base = workspace_root() if target.startswith("data set/") else PACKAGE_ROOT
        return base / target
    return Path(value)
