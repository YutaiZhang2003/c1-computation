"""Paths and source inventory for the retained genus-one three-point code."""
from pathlib import Path
import json
import os
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'PACKAGE_MAP.json').is_file())
LAYOUT = ROOT / 'code/one_loop/LAYOUT.json'

def source_path(value):
    """Resolve a historical live-source path, leaving data paths untouched."""
    p = Path(value)
    key = str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)
    layout = json.loads(LAYOUT.read_text())
    target = layout['path_map'].get(key, key)
    return ROOT / target if not Path(target).is_absolute() else Path(target)

def install():
    """Expose retained flat modules and their shared dependencies."""
    layout = json.loads(LAYOUT.read_text())
    paths = {str((ROOT / p).parent) for p in layout['module_map'].values()
             if '/archive/' not in p}
    paths.update(str((ROOT / p).parent) for p in layout['shared_sources'] if p.endswith('.py'))
    paths.add(str(ROOT / 'code/one_loop'))
    paths.add(str(ROOT / 'code/general/io'))
    for path in sorted(paths, reverse=True):
        if path not in sys.path: sys.path.insert(0, path)
    sys.dont_write_bytecode = True
    # Direct worker subprocesses inherit the same local source roots.
    previous = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    os.environ['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(sorted(paths) + [p for p in previous if p]))
    return ROOT

def production_sources():
    """All retained source inputs, excluding numerical data and archives."""
    layout = json.loads(LAYOUT.read_text())
    names = set(layout['retained_sources']) | set(layout['shared_sources'])
    names.update(('PACKAGE_MAP.json', 'run.py', 'code/one_loop/LAYOUT.json',
                  'code/tree_level/LAYOUT.json',
                  'code/tree_level/general/tree_level_source_compatibility.py'))
    vendor = ROOT / 'code/general/virasoro/vendor/genus0_elliptic_h_recursion/src'
    names.update(str(p.relative_to(ROOT)) for p in vendor.rglob('*.py'))
    paths = sorted(ROOT / name for name in names)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError('Missing retained source: ' + ', '.join(missing))
    return paths
