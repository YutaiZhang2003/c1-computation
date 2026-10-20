"""Validate the explicit production subset of an original frozen bundle."""
import json
from pathlib import Path
import hashlib

def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def verify_bundle(run):
    run=Path(run).resolve()
    manifest=json.loads((run/'BUNDLE.json').read_text())
    if object_sha(manifest['files']) != manifest['bundle_sha256']:
        raise ValueError('original bundle fingerprint mismatch')
    selection=json.loads((run/'PACKAGED_INPUTS.json').read_text())
    if selection['source_bundle_sha256'] != manifest['bundle_sha256']:
        raise ValueError('wrong packaged bundle identity')
    original={e['path']:e['sha256'] for e in manifest['files']}
    for e in selection['files']:
        path=(run/e['path']).resolve()
        if not path.is_relative_to(run) or original.get(e['path'])!=e['sha256'] or hashlib.sha256(path.read_bytes()).hexdigest()!=e['sha256']:
            raise ValueError('packaged production input changed: '+e['path'])
    return manifest
