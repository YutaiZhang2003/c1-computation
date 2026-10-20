"""Resolve historical module names and verify recorded source-only reorganizations."""
from pathlib import Path
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import sys


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical, symbols=None):
        self.canonical = canonical
        self.symbols = symbols

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self.symbols:
            # Merged modules can have different public names or policies.
            # Resolve each recorded export to its actual implementation.
            for name, (target, attribute) in self.symbols.items():
                value = importlib.import_module(target)
                for part in attribute.split('.'):
                    value = getattr(value, part)
                setattr(module, name, value)
            canonical = importlib.import_module(self.canonical)
            module.__file__ = canonical.__file__
            module.__getattr__ = lambda name: getattr(canonical, name)
            return
        # Both names refer to one module, preserving classes and module state.
        sys.modules[module.__name__] = importlib.import_module(self.canonical)


class _LegacyModuleNames(importlib.abc.MetaPathFinder):
    _tree_level_name_aliases = True

    def __init__(self, aliases, symbols=None):
        self.aliases = dict(aliases)
        self.symbols = symbols or {}

    def find_spec(self, fullname, path=None, target=None):
        canonical = self.aliases.get(fullname)
        if canonical is not None:
            return importlib.util.spec_from_loader(
                fullname, _AliasLoader(canonical, self.symbols.get(fullname)))
        return None


def install_module_aliases(aliases, symbols=None):
    """Use aliases only after normal import lookup, including frozen source files."""
    sys.meta_path[:] = [finder for finder in sys.meta_path
                       if not getattr(finder, '_tree_level_name_aliases', False)]
    sys.meta_path.append(_LegacyModuleNames(aliases, symbols))


def source_matches_reference(root, relative, reference):
    """Accept exact bytes or a recorded source revision ending at the current hash.

    A later source edit invalidates a recorded pair. Old frozen bundles keep
    their own filenames and source bytes; this function only checks the live
    source when preparing a new bundle from an accepted one.
    """
    root, relative, reference = Path(root), Path(relative).as_posix(), Path(reference)
    layout = json.loads((root / 'code/tree_level/LAYOUT.json').read_text())
    destination = layout['path_map'].get(relative, relative)
    current = root / destination
    if not current.is_file():
        return False
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    current_hash, reference_hash = sha(current), sha(reference)
    if current_hash == reference_hash:
        return True
    records = (layout['files'] + layout['support_files']
               + layout['new_helpers'] + layout['campaign_copies'])
    if relative == 'code/tree_level/LAYOUT.json':
        # A manifest cannot contain its own hash. Permit a pinned prior manifest
        # only while every recorded current source still has its expected hash.
        return (reference_hash in layout.get('prior_layout_sha256', [])
                and all((root / r['path']).is_file()
                        and sha(root / r['path']) == r['sha256'] for r in records))
    for record in records:
        if record['path'] != destination:
            continue
        revisions = record.get('source_revisions', [])
        if any(r['path'] == relative and r['sha256'] == reference_hash for r in revisions):
            return current_hash == record['sha256']
    return False
