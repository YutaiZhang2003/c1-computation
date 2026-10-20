"""Selective block refinement, immutable records, and paired Monte Carlo statistics."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def seal(value):
    if 'fingerprint' in value:
        raise ValueError('already sealed')
    return dict(value, fingerprint=json_hash(value))


def read_manifest(path):
    value = json.loads(Path(path).read_text())
    if value.get('fingerprint') != json_hash({k: v for k, v in value.items() if k != 'fingerprint'}):
        raise ValueError('manifest fingerprint mismatch: ' + str(path))
    return value


def write_json_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('immutable JSON already differs: ' + str(path))
        return
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def save_arrays(path, arrays, metadata, *, checkpoint=False):
    """Complete files are immutable; checkpoints only accumulate accepted rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {k: array_hash(v) for k, v in arrays.items()}
    meta = seal(dict(metadata, array_hashes=identity))
    if path.exists() and not checkpoint:
        old_meta, _ = load_arrays(path)
        if old_meta != meta:
            raise ValueError('immutable arrays already differ: ' + str(path))
        return meta
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            np.savez_compressed(f, metadata=json.dumps(meta, sort_keys=True, allow_nan=False), **arrays)
        if checkpoint:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return meta


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z['metadata'].item())
        arrays = {k: z[k] for k in z.files if k != 'metadata'}
    if (meta.get('fingerprint') != json_hash({k: v for k, v in meta.items() if k != 'fingerprint'})
            or meta.get('array_hashes') != {k: array_hash(v) for k, v in arrays.items()}):
        raise ValueError('array shard content/identity mismatch: ' + str(path))
    return meta, arrays


def degree_keys(order, minimum=0):
    if type(order) is not int or type(minimum) is not int or not 0 <= minimum <= order:
        raise ValueError('invalid total degree')
    return np.asarray(sorted(((a, b, c) for a in range(order + 1)
        for b in range(order + 1 - a) for c in range(order + 1 - a - b)
        if a + b + c >= minimum), key=lambda k: (sum(k), *k)), dtype=np.int64)


def point_slots(replicate_ids, point_indices):
    reps, points = np.asarray(replicate_ids), np.asarray(point_indices)
    if reps.shape != (8,) or points.shape != (8, 256):
        raise ValueError('original eight by 256 sample design required')
    if reps.dtype.kind not in 'iu' or points.dtype.kind not in 'iu':
        raise ValueError('integer original IDs required')
    mapping = {(int(reps[r]), int(points[r, n])): (r, n) for r in range(8) for n in range(256)}
    if len(mapping) != 2048 or set(mapping) != {(r, n) for r in range(8) for n in range(256)}:
        raise ValueError('missing, duplicated, or unexpected original source IDs')
    return mapping


def require_active(status, order):
    shell_keys(order)
    if status != 'needs_' + str(order):
        raise ValueError('point already stopped or stage is out of sequence')


def contract_chiral(coefficients, keys, q):
    """C(q) and C(qbar) use the SAME, potentially complex coefficient table."""
    q = np.asarray(q, dtype=complex)
    if q.shape != (3,) or not np.isfinite(q).all() or np.any(abs(q) == 0):
        raise ValueError('finite nonzero q triple required')
    monomials = np.prod(q[None, :] ** np.asarray(keys), axis=1)
    result = np.asarray(coefficients) @ np.column_stack((monomials, monomials.conjugate()))
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite chiral contraction')
    return result


def apply_corrections(values, slots, corrections):
    old = np.asarray(values)
    if old.shape[:2] != (8, 256) or len(slots) != len(corrections):
        raise ValueError('original full point array and one correction per target required')
    if len(set(map(tuple, slots))) != len(slots) or not np.isfinite(corrections).all():
        raise ValueError('duplicated target or numerical failure')
    updated = old.copy()
    delta = np.zeros((8, 256), dtype=complex)
    for (r, n), change in zip(slots, corrections):
        if not 0 <= r < 8 or not 0 <= n < 256:
            raise ValueError('target slot outside original design')
        updated[r, n] += change
        delta[r, n] = change
    return updated, delta


def paired_statistics(original_scramble_means, delta_points):
    old, delta = np.asarray(original_scramble_means, complex), np.asarray(delta_points, complex)
    if old.shape != (8,) or delta.shape != (8, 256) or not np.isfinite(old).all() or not np.isfinite(delta).all():
        raise ValueError('finite eight-scramble full-sample inputs required')
    shifts = delta.sum(axis=1) / 256
    refined = old + shifts
    def stats(x):
        return dict(mean=[float(x.real.mean()), float(x.imag.mean())],
                    se=[float(x.real.std(ddof=1) / np.sqrt(8)), float(x.imag.std(ddof=1) / np.sqrt(8))])
    return dict(original=stats(old), refined=stats(refined), shift=stats(shifts)), refined, shifts


class RefinementPolicy:
    """One frozen order range and its relative block-truncation criterion."""
    def __init__(self, version, tolerance, base_order, cap, statuses):
        self.VERSION = version
        self.TOLERANCE = tolerance
        self.BASE_ORDER = base_order
        self.CAP = cap
        self.STATUSES = statuses

    def shell_keys(self, order):
        if type(order) is not int or not self.BASE_ORDER < order <= self.CAP:
            raise ValueError('only missing degrees 13--16 may be generated')
        return degree_keys(order, order)

    def validate_shell(self, keys, coefficients, order, row_count):
        if not np.array_equal(keys, self.shell_keys(order)):
            raise ValueError('shell degree coverage or key order differs')
        c = np.asarray(coefficients)
        if c.dtype != np.complex128 or c.shape != (row_count, len(keys)) or not np.isfinite(c).all():
            raise ValueError('shell coefficients lack finite exact coverage')

    def merge_overlay(self, parent, shells):
        """Validate a complete in-memory prefix without emitting parent outputs."""
        parent = np.asarray(parent)
        if parent.ndim != 2 or parent.shape[1] != len(degree_keys(12)) or not np.isfinite(parent).all():
            raise ValueError('complete finite C12 parent required')
        chunks = [parent]
        for order, keys, coefficients in shells:
            if order != 12 + len(chunks):
                raise ValueError('overlay skipped or duplicated a degree')
            self.validate_shell(keys, coefficients, order, len(parent))
            chunks.append(coefficients)
        return np.concatenate(chunks, axis=1)

    def gaps_pass(self, numerators, denominator, highest_value):
        num = np.asarray(numerators, dtype=float)
        den = np.asarray(denominator, dtype=float)
        total = np.asarray(highest_value, dtype=complex)
        if num.shape != (*den.shape, 2) or total.shape != den.shape:
            raise ValueError('both two-order gaps and complete highest sum required')
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            relative = num / den[..., None]
        passed = ((den > 0) & np.isfinite(den) & np.isfinite(total)
                  & np.isfinite(num).all(axis=-1) & (num >= 0).all(axis=-1)
                  & np.isfinite(relative).all(axis=-1) & (relative < self.TOLERANCE).all(axis=-1))
        return passed, relative

    def classify(self, order, passed):
        if type(order) is not int or order not in range(12, 17):
            raise ValueError('candidate outside C12--C16')
        if passed:
            return self.STATUSES[0] if order == 12 else 'converged_at_' + str(order)
        return 'cap_reached_unconverged' if order == self.CAP else 'needs_' + str(order + 1)

    def nonchiral_diagnostic(self, chirals, logarithms, order):
        orders = (order - 4, order - 2, order)
        if set(orders) - chirals.keys():
            raise ValueError('missing matched two-order prefix')
        with np.errstate(over='raise', invalid='raise'):
            weights = np.exp(logarithms)
            absolute_weights = np.exp(np.asarray(logarithms).real)
            products = [chirals[n][:, 0] * chirals[n][:, 1] for n in orders]
            totals = [np.sum(weights * chirals[n][:, 0] * chirals[n][:, 1]) for n in orders]
            numerators = np.asarray([np.sum(absolute_weights * abs(products[j + 1] - products[j]))
                                     for j in range(2)])
        if not all(np.isfinite(p).all() for p in products) or not np.isfinite(totals).all():
            raise FloatingPointError('nonfinite matched-prefix spectral sum')
        denominator = abs(totals[-1])
        passed, relative = self.gaps_pass(numerators, np.asarray(denominator), np.asarray(totals[-1]))
        return dict(orders=orders, totals=np.asarray(totals), numerators=numerators,
                    denominator=denominator, relative=relative, passed=bool(passed))


class ContinuationPolicy(RefinementPolicy):
    def shell_keys(self, order):
        if type(order) is not int or not self.BASE_ORDER < order <= self.CAP:
            raise ValueError('only missing degrees 15--18 may be generated')
        return degree_keys(order, order)

    def classify(self, order, passed):
        if type(order) is not int or order not in range(15, 19):
            raise ValueError('candidate outside C15--C18')
        if passed:
            return 'converged_at_' + str(order)
        return 'cap_reached_unconverged' if order == self.CAP else 'needs_' + str(order + 1)


initial = RefinementPolicy('sphere-six-point-selective-c16-v1', 0.005, 12, 16, ('passes_at_12_under_0p5pct', 'converged_at_13', 'converged_at_14', 'converged_at_15', 'converged_at_16', 'cap_reached_unconverged'))
VERSION = initial.VERSION
TOLERANCE = initial.TOLERANCE
BASE_ORDER = initial.BASE_ORDER
CAP = initial.CAP
STATUSES = initial.STATUSES
shell_keys = initial.shell_keys
validate_shell = initial.validate_shell
merge_overlay = initial.merge_overlay
gaps_pass = initial.gaps_pass
classify = initial.classify
nonchiral_diagnostic = initial.nonchiral_diagnostic
continuation = ContinuationPolicy('sphere-six-point-selective-2pct-c18-v1', 0.02, 14, 18, ('passes_at_12_under_0p5pct', 'converged_at_13', 'converged_at_14', 'passes_at_14_under_2pct', 'converged_at_15', 'converged_at_16', 'converged_at_17', 'converged_at_18', 'cap_reached_unconverged'))
c18_VERSION = continuation.VERSION
c18_TOLERANCE = continuation.TOLERANCE
c18_BASE_ORDER = continuation.BASE_ORDER
c18_CAP = continuation.CAP
c18_STATUSES = continuation.STATUSES
c18_shell_keys = continuation.shell_keys
c18_validate_shell = continuation.validate_shell
c18_gaps_pass = continuation.gaps_pass
c18_classify = continuation.classify
c18_nonchiral_diagnostic = continuation.nonchiral_diagnostic
