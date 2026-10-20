"""Preparation, integrity checks, and evaluation of saved native block banks."""
from __future__ import annotations

# genus1_shared_native_bank
# Reusable full necklace/pair/comb CFT on an exactly reweighted proposal.
# 
# This is a bank and pointwise validation layer, not a production region
# certificate. It never builds coefficients during an evaluation, interpolates
# in momentum, transforms native q, or imports an old amplitude. Wider regional
# coverage, independent spectral checks and collar/necklace adapters are required
# before the full moduli integrator may use it.



import hashlib
import json
import math
import cmath
import os
import tempfile
from pathlib import Path

import numpy as np

from genus1_rebuild_policy import require_orders

FORMAT = 'genus1-shared-native-bank-v1'
NUMERICS = 'mp-ccy-verified-v1'
BASIS = 'native-sewing-v2'


def atomic_create(path,write):
    """Publish a fully written file without replacing an existing artifact."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.publish-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:write(f);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)
    finally:Path(temporary).unlink(missing_ok=True)


def freeze_arrays(path,metadata,arrays):
    """Data first, receipt as commit marker; recover only identical orphan data."""
    path=Path(path);receipt=path.with_suffix('.json')
    if receipt.exists():raise FileExistsError('preserve existing bank; choose a new output path')
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays) or any(old[k].dtype!=np.asarray(v).dtype or
                    not np.array_equal(old[k],v) for k,v in arrays.items()):
                raise ValueError('orphan bank data differs from verified preparation')
    else:atomic_create(path,lambda f:np.savez_compressed(f,**arrays))
    metadata=dict(metadata,data_sha256=digest(path))
    blob=(json.dumps(metadata,indent=2,allow_nan=False)+'\n').encode()
    atomic_create(receipt,lambda f:f.write(blob))
    return metadata


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(kin):
    def numbers(values):
        return [[float(complex(v).real),float(complex(v).imag)] for v in values]
    if kin['coefficient_basis'] != BASIS or kin['channel'] not in ('pair_ope','comb_ope','necklace'):
        raise ValueError('only full native necklace/pair/comb CFT is supported by this bank')
    return dict(coefficient_basis=BASIS,coefficient_numerics=NUMERICS,central_charge=25.,
        channel=kin['channel'],external_weights=numbers(kin['external_weights']),
        external_momenta=numbers(kin['external_momenta']))


def point_kinematics(atlas, point):
    """The same reduced CFT convention as reduced_channel_integrand()."""
    from genus1_three_point_momentum import geometry
    w1,w2,tau = (complex(*point[k]) for k in ('w1','w2','tau'))
    if point['channel'] != 'necklace':
        return geometry(atlas,w1,w2,tau,tuple(point['pair']),point['channel'])
    from genus1_three_point_worldsheet import ordered_necklace_data
    from genus1_sewing_domains import necklace_chart
    from genus1_three_point_collision import _common_factor, _exact_koba
    neck = atlas.necklace
    # Existing necklace ordering uses the equal outgoing weights on this
    # 1->2 slice. Do not silently extend it to unequal external insertions.
    if neck.external_momenta[0] != neck.external_momenta[1]:
        raise ValueError('necklace adapter currently requires equal outgoing momenta')
    _,logs = ordered_necklace_data(w1,w2,tau)
    necklace_chart(logs).require()
    return dict(coefficient_basis=BASIS,channel='necklace',logs=logs,
        variables=tuple(cmath.exp(z) for z in logs),decay=np.asarray([-2*z.real for z in logs]),
        external_weights=neck.external_weights,external_momenta=neck.external_momenta,
        constant=0.,common_log=cmath.log(_common_factor(atlas,tau)*_exact_koba(atlas,w1,w2,tau)))


def graph_structure_log(special, kin, momenta):
    """Bank construction only; this is never called during bank evaluation."""
    if kin['channel'] in ('pair_ope','comb_ope'):
        from genus1_three_point_momentum import structure_log
        return structure_log(special,kin,momenta)
    if kin['channel'] != 'necklace':
        raise ValueError('unsupported CFT graph')
    from liouville_torus import log_yin_structure_constant_momentum
    return sum((complex(log_yin_structure_constant_momentum(special,
        kin['external_momenta'][i],momenta[i],momenta[(i+1)%3])) for i in range(3)),0j)


def require_storage_orders(orders, metadata):
    """Keep the campaign cap; admit only the adaptive narrow-necklace policy."""
    try:
        return require_orders(orders)
    except ValueError:
        if (metadata.get('pointwise_order_policy')=='uniform-through-16-v1' and
                len(orders)==3 and all(not isinstance(n,bool) and math.isfinite(n)
                and int(n)==n and 0<=n<=16 for n in orders)):
            return list(map(int,orders))
        if (metadata.get('adaptive_bank_version') != 'genus1-adaptive-cft-v2'
                or metadata.get('identity', {}).get('channel') != 'necklace'
                or len(orders) != 3
                or any(isinstance(n, bool) or not math.isfinite(n) or int(n) != n
                       or n < 0 or n > 24 for n in orders)
                or sorted(orders)[1] > 2):
            raise ValueError('orders above 10 require an adaptive necklace with '
                             'one axis <=24 and the other two <=2') from None
        return list(map(int, orders))


def variants(orders, *, validate=require_orders):
    orders = validate(orders)
    cuts = [('full',tuple(orders))]
    for axis,n in enumerate(orders):
        step = 2 if n >= 6 else 1
        for cut in (max(0,n-step),max(0,n-2*step)):
            prefix = list(orders); prefix[axis] = cut
            cuts.append((f'axis{axis}_{cut}',tuple(prefix)))
    return cuts


class SharedNativeBank:
    """Geometry-independent saved CFT inputs; every primary factor stays exact."""
    def __init__(self, metadata, arrays, *, expected_source_sha256):
        if metadata.get('format') != FORMAT or metadata.get('target_blind') is not True:
            raise ValueError('incompatible or non-blind coefficient bank')
        if (metadata.get('source_manifest_sha256') != expected_source_sha256 or
                len(expected_source_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in expected_source_sha256)):
            raise ValueError('unverified CFT source fingerprint')
        key = metadata['identity']
        if (key.get('coefficient_basis') != BASIS or key.get('coefficient_numerics') != NUMERICS or
                key.get('central_charge') != 25. or key.get('channel') not in ('pair_ope','comb_ope','necklace')):
            raise ValueError('legacy/unsupported CFT coefficients')
        self.orders = tuple(require_storage_orders(metadata['orders'], metadata))
        degree = metadata.get('coefficient_total_degree')
        if degree is not None and (key['channel'] not in ('pair_ope', 'comb_ope') or
                isinstance(degree, bool) or not isinstance(degree, int) or
                degree < 0 or degree > sum(self.orders)):
            raise ValueError('invalid sparse native coefficient region')
        self.coefficient_total_degree = degree
        self.metadata = json.loads(json.dumps(metadata,allow_nan=False))
        self.arrays = {key:np.asarray(arrays[key]).copy() for key in
            ('momenta','log_weights','structure_logs','coefficients','coefficient_precision_bounds')}
        n = len(self.arrays['momenta'])
        if not n:
            raise ValueError('empty momentum bank')
        shapes = dict(momenta=(n,3),log_weights=(n,),structure_logs=(n,),
            coefficients=(n,*[o+1 for o in self.orders]),coefficient_precision_bounds=(n,))
        for key,arr in self.arrays.items():
            if arr.shape != shapes[key] or not np.all(np.isfinite(arr)):
                raise ValueError('invalid bank array: '+key)
            arr.flags.writeable = False
        if degree is not None:
            outside = np.fromfunction(lambda a,b,c: a+b+c > degree,
                                      tuple(o+1 for o in self.orders), dtype=int)
            if np.any(self.arrays['coefficients'][:, outside] != 0):
                raise ValueError('sparse bank has nonzero coefficients outside its declared region')
        if np.any(self.arrays['momenta'] <= 0):
            raise ValueError('positive real Liouville momentum nodes required')
        errors = self.arrays['coefficient_precision_bounds']
        if np.any(errors < 0) or np.any(errors > 2e-11+2e-13):
            raise ValueError('coefficient precision check failed')
        if metadata.get('quadrature',{}).get('measure') != 'product_dP':
            raise ValueError('bank weights must have the unchanged dP measure')

    def validate_orders(self, orders):
        return require_storage_orders(orders, self.metadata)

    def covers(self, orders, total_degree=None):
        """Whether every coefficient needed by this finite definition exists."""
        if any(a > b for a,b in zip(orders, self.orders)):
            return False
        if self.coefficient_total_degree is None:
            return True
        required = sum(orders) if total_degree is None else min(total_degree, sum(orders))
        return required <= self.coefficient_total_degree

    def evaluate_probes(self, atlas, points, *, orders=None):
        """Evaluate full reduced CFT and per-axis prefixes; not a coverage claim.

        Each point supplies w1, w2, tau as complex pairs, pair and channel.
        geometry() enforces the actual native sewing domain at every call.
        Compatible external momenta/weights and graph are required exactly.
        Momentum nodes and the full DOZZ values are reused without change.
        """
        requested = self.orders if orders is None else tuple(self.validate_orders(orders))
        if not self.covers(requested):
            raise ValueError('missing higher-order coefficients; compute and check overlaps first')
        cuts = variants(requested, validate=self.validate_orders)
        answers = []
        ps2 = self.arrays['momenta']**2
        for point in points:
            kin = point_kinematics(atlas,point)
            if identity(kin) != self.metadata['identity']:
                raise ValueError('different channel or external frequencies: bank reuse forbidden')
            powers = [z**np.arange(n+1) for z,n in zip(kin['variables'],requested)]
            base = (self.arrays['structure_logs'] + kin['constant'] + kin['common_log']
                    - np.einsum('ti,i->t',ps2,kin['decay']) - 3*math.log(math.pi)
                    + self.arrays['log_weights'])
            row = []
            for _,cut in cuts:
                block = np.einsum('tijk,i,j,k->t',
                    self.arrays['coefficients'][(slice(None),*[slice(n+1) for n in cut])],
                    *(power[:n+1] for power,n in zip(powers,cut)),optimize=False)
                if not np.all(np.isfinite(block)):
                    raise FloatingPointError('nonfinite native-q block')
                values = np.zeros(len(block),complex)
                nonzero = block != 0
                logvalue = base[nonzero] + 2*np.log(abs(block[nonzero]))
                if np.any(logvalue.real > math.log(np.finfo(float).max)):
                    raise FloatingPointError('weighted CFT density overflow')
                values[nonzero] = np.exp(logvalue)
                if not np.all(np.isfinite(values)):
                    raise FloatingPointError('nonfinite weighted CFT value')
                row.append(complex(math.fsum(values.real),math.fsum(values.imag)))
            answers.append(row)
        return np.asarray(answers,complex)

    def require_coefficient_overlap(self, higher, *, rtol=2e-11, atol=2e-13):
        """Verify newly computed coefficients against this bank on identical nodes."""
        for key in ('identity','source_manifest_sha256','quadrature'):
            if self.metadata[key] != higher.metadata[key]:
                raise ValueError('incompatible higher-order bank identity')
        for key in ('momenta','log_weights','structure_logs'):
            if not np.array_equal(self.arrays[key],higher.arrays[key]):
                raise ValueError('higher-order overlap requires identical '+key)
        if any(a > b for a,b in zip(self.orders,higher.orders)):
            raise ValueError('the new bank does not contain the old tensor prefix')
        previous = self.arrays['coefficients']
        prefix = higher.arrays['coefficients'][(slice(None),*[slice(n+1) for n in self.orders])]
        cut = min(self.coefficient_total_degree if self.coefficient_total_degree is not None else sum(self.orders),
                  higher.coefficient_total_degree if higher.coefficient_total_degree is not None else sum(higher.orders))
        overlap = np.fromfunction(lambda a,b,c: a+b+c <= cut,
                                  tuple(o+1 for o in self.orders), dtype=int)
        a = previous[:, overlap]; b = prefix[:, overlap]
        delta = abs(a-b)
        if not np.all(delta <= atol+rtol*np.maximum(abs(a),abs(b))):
            raise ValueError('overlapping coefficients disagree')
        return float(np.max(delta/np.maximum(1.,np.maximum(abs(a),abs(b)))))

    def freeze(self, path):
        """New immutable artifact with a hash-linked metadata receipt; no pickle."""
        path = Path(path)
        receipt = path.with_suffix('.json')
        metadata = freeze_arrays(path,self.metadata,self.arrays)
        return dict(data_sha256=metadata['data_sha256'],receipt_sha256=digest(receipt))

    @classmethod
    def load(cls, path, *, expected_source_sha256, expected_receipt_sha256):
        path = Path(path); receipt = path.with_suffix('.json')
        if digest(receipt) != expected_receipt_sha256:
            raise ValueError('bank receipt checksum mismatch')
        meta = json.loads(receipt.read_text())
        if digest(path) != meta['data_sha256']:
            raise ValueError('bank data checksum mismatch')
        with np.load(path,allow_pickle=False) as data:
            return cls(meta,data,expected_source_sha256=expected_source_sha256)


# genus1_shared_bank_ensemble
# Runtime full/prefix refinement checks on saved native-q CFT banks.
# 
# Every new geometry is checked with three momentum orders and an independent
# proposal width. Saved probe acceptance is never inherited. This is not a
# complete region catalogue or an analytic-collar/forest amplitude adapter.



from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path

import json
from pathlib import Path

import numpy as np

from genus1_rebuild_policy import (REFINEMENT_RTOL, classify_cft_point,
                                  weighted_cft_record, encoded)


class UnresolvedSharedBankPoint(ValueError):
    def __init__(self, diagnostic):
        super().__init__('shared bank point needs refinement: '+diagnostic['classification']['status'])
        self.diagnostic = diagnostic


class SharedBankEnsemble:
    """Three increasing dP quadratures plus the finest width-check rule."""
    def __init__(self, banks):
        if len(banks) != 4:
            raise ValueError('three refinement rules and one width rule are required')
        self.banks = tuple(banks)
        reference = banks[0].metadata
        for bank in banks:
            for key in ('identity', 'orders', 'source_manifest_sha256', 'design_sha256'):
                if bank.metadata.get(key) != reference.get(key):
                    raise ValueError('incompatible bank ensemble: '+key)
        quadratures = [bank.metadata['quadrature'] for bank in banks]
        def proposal(q):
            return {k:v for k,v in q.items() if k not in ('nodes_per_edge','scale')}
        if any(proposal(q) != proposal(quadratures[0]) for q in quadratures[1:]):
            raise ValueError('refinements must share a proposal definition')
        ns = [q['nodes_per_edge'] for q in quadratures]
        if (any(type(n) is not int or n <= 0 for n in ns) or
                not ns[0] < ns[1] < ns[2] or ns[3] != ns[2]):
            raise ValueError('non-increasing or incomplete momentum refinement')
        if [q['scale'] for q in quadratures] != [1.,1.,1.,.8]:
            raise ValueError('the independent 0.8-width check must be preserved')
        if any(len(b.arrays['momenta']) != n**3 for b,n in zip(banks,ns)):
            raise ValueError('momentum tensor node count mismatch')
        self.orders = banks[0].orders
        self.node_orders = ns

    @classmethod
    def load(cls, folder, *, expected_export_sha256, expected_source_sha256):
        folder = Path(folder)
        path = folder/'export_report.json'
        if digest(path) != expected_export_sha256:
            raise ValueError('unverified bank export report')
        report = json.loads(path.read_text())
        if (report['status'] != 'passed_saved_probe_bank_reuse' or
                report['target_blind'] is not True or
                report['manifest_sha256'] != expected_source_sha256):
            raise ValueError('incompatible bank export')
        # Replay evidence is tied to the actual bank implementation, not just
        # to a user-editable success flag in an export report.
        implementation = _source_path(Path('code/one_loop/production') / 'genus1_shared_native_bank.py')
        if report['export_source_sha256'].get(implementation.name) != digest(implementation):
            raise ValueError('bank implementation changed since export validation')
        banks = []
        for row in report['banks']:
            name = row['path']
            if Path(name).name != name:
                raise ValueError('bank export paths must be local basenames')
            banks.append(SharedNativeBank.load(folder/name,
                expected_source_sha256=expected_source_sha256,
                expected_receipt_sha256=row['receipt_sha256']))
        return cls(banks)

    def audit(self, atlas, point):
        """Fresh empirical checks; invalid native charts raise, never fall back."""
        values = np.asarray([b.evaluate_probes(atlas,[point])[0] for b in self.banks])
        if values.shape != (4,7) or not np.all(np.isfinite(values)):
            raise FloatingPointError('invalid full/prefix ensemble values')
        if np.any(abs(values[:,0]) < np.finfo(float).tiny):
            raise FloatingPointError('vanishing/underflowed full CFT cannot certify refinement')
        changes = [abs(values[1]-values[0])/np.maximum(abs(values[1]),1e-300),
                   abs(values[2]-values[1])/np.maximum(abs(values[2]),1e-300),
                   abs(values[3]-values[2])/np.maximum(abs(values[2]),1e-300)]
        momentum_passed = bool(all(np.all(c <= REFINEMENT_RTOL) for c in changes))
        full = values[2,0]
        axis_changes = []
        for axis in range(3):
            first,second = values[2,1+2*axis:3+2*axis]
            axis_changes.append([float(abs(full-first)/abs(full)),
                                 float(abs(first-second)/abs(full))])
        classification = classify_cft_point(orders=self.orders,axis_changes=axis_changes,
            chart_admissible=True,coefficients_verified=True,momentum_converged=momentum_passed)
        return dict(point=point,classification=classification,value=encoded(full),
            momentum_node_orders=self.node_orders,
            momentum_full_and_prefix_relative_changes=[c.tolist() for c in changes],
            maximum_momentum_relative_changes=[float(max(c)) for c in changes],
            prefixes={label:encoded(z) for (label,_),z in zip(variants(self.orders),values[2])},
            amplitude_contribution=None,coefficient_recomputations=0,dozz_recomputations=0,
            target_blind=True,full_amplitude_certified=False,
            qualification='Fresh pointwise empirical refinement, not a rigorous remainder bound or full regional coverage.')

    def require(self, atlas, point):
        result = self.audit(atlas,point)
        if not result['classification']['include_in_amplitude']:
            raise UnresolvedSharedBankPoint(result)
        return result

    def weighted_record(self, atlas, point, *, weight, point_id):
        """Caller must supply the true normalized MC weight, never a pilot weight."""
        result = self.require(atlas,point)
        classification = result['classification']
        prefixes = {str(i):complex(*result['prefixes'][f'axis{i}_8'])
                    for i in classification['capped_axes']}
        return weighted_cft_record(classification,value=complex(*result['value']),
            weight=weight,cap_prefix_values=prefixes,point_id=point_id)


# genus1_cusp_banks
# Restartable, hash-checked coefficient catalogs for the NEW cusp pilot.
# 
# The order-10 campaign bank/policy is untouched. This separate format permits
# the reviewed higher-order checks and carries no inherited convergence flag.
# Sampling opens completed coefficient chunks read-only; only build() computes
# DOZZ/CCY coefficients. Each exact graph+momentum tuple is built once at the
# largest tensor requested by the frozen catalog.



from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile
import numpy as np

from genus1_source_paths import ROOT
from genus1_moduli_sampling import integer
from genus1_three_point_momentum import fingerprint
CUSP_BANK_FORMAT = 'genus1-cusp-native-bank-v1'


def encode_cusp_value(value):
    if isinstance(value,complex): return [value.real,value.imag]
    if isinstance(value,np.ndarray): return encode_cusp_value(value.tolist())
    if isinstance(value,np.generic): return encode_cusp_value(value.item())
    if isinstance(value,dict): return {str(k):encode_cusp_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [encode_cusp_value(v) for v in value]
    return value


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    checksum=path.with_suffix(path.suffix+'.sha256')
    if checksum.exists():
        raise FileExistsError('preserve existing pilot artifact: '+str(path))
    blob=(json.dumps(encode_cusp_value(value),indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=blob:raise ValueError('orphan JSON differs from verified preparation: '+str(path))
    else:atomic_create(path,lambda f:f.write(blob))
    sha=hashlib.sha256(blob).hexdigest()
    atomic_create(checksum,lambda f:f.write((sha+'\n').encode()))
    return sha


def read_json(path):
    path=Path(path)
    if digest(path) != path.with_suffix(path.suffix+'.sha256').read_text().strip():
        raise ValueError('changed pilot JSON: '+str(path))
    return json.loads(path.read_text())


def source_manifest():
    from genus1_source_paths import production_sources
    return {str(p.relative_to(ROOT)): digest(p) for p in production_sources()}


def check_sources(plan):
    for name,sha in plan['source_files'].items():
        p=(ROOT/name).resolve()
        if ROOT not in p.parents or digest(p)!=sha:
            raise ValueError('pilot source changed; prepare a new plan: '+name)


def decode_identity(key):
    return dict(channel=key['channel'],coefficient_basis=key['coefficient_basis'],
        external_weights=tuple(complex(*x) for x in key['external_weights']),
        external_momenta=tuple(complex(*x) for x in key['external_momenta']))


def chunk_nodes(plan):
    size=integer(plan['coefficient_chunk_size'],'chunk size',1)
    groups={}
    for key,spec in plan['catalog']['nodes'].items():
        group=(fingerprint(spec['identity']),tuple(spec['orders']))
        groups.setdefault(group,[]).append(key)
    result=[]
    for group in sorted(groups):
        ids=sorted(groups[group])
        result.extend(ids[i:i+size] for i in range(0,len(ids),size))
    return result


def load_chunk(root, index, plan_sha, ids):
    folder=Path(root)/'banks'/f'{index:07d}'
    receipt=read_json(folder/'receipt.json')
    if (receipt['format']!=CUSP_BANK_FORMAT or receipt['plan_sha256']!=plan_sha or
            receipt['node_ids']!=ids or digest(folder/'data.npz')!=receipt['data_sha256']):
        raise ValueError('coefficient chunk identity/hash mismatch')
    with np.load(folder/'data.npz',allow_pickle=False) as f:
        arrays={k:f[k] for k in f}
    if any(not np.all(np.isfinite(a)) for a in arrays.values()):
        raise FloatingPointError('nonfinite saved CFT chunk')
    return receipt,arrays


def build(root, *, start=0, stop=None, reuse_root=None, progress=None):
    """Explicit construction stage; resumable by complete atomic chunks."""
    from liouville_torus import UpsilonB
    from torus_c_recursion_numerics import stable_c_recursion_coefficients
    root=Path(root);plan=read_json(root/'plan.json');check_sources(plan)
    plan_sha=digest(root/'plan.json');chunks=chunk_nodes(plan)
    start=integer(start,'start');stop=len(chunks) if stop is None else integer(stop,'stop')
    if not 0 <= start <= stop <= len(chunks): raise ValueError('invalid chunk interval')
    special=UpsilonB(1.,dps=plan['special_dps'])
    donor = None if reuse_root is None else BankResolver(reuse_root)
    if donor is not None and donor.plan['source_files'] != plan['source_files']:
        raise ValueError('reuse requires the same verified scientific implementation')
    for index in range(start,stop):
        ids=chunks[index];folder=root/'banks'/f'{index:07d}'
        if folder.exists():
            load_chunk(root,index,plan_sha,ids)
            if progress: progress(dict(stage='reused_chunk',chunk=index))
            continue
        specs=[plan['catalog']['nodes'][key] for key in ids]
        maximum=tuple(max(row['orders'][a] for row in specs) for a in range(3))
        coeff=np.zeros((len(ids),*[n+1 for n in maximum]),complex)
        arrays=dict(momenta=np.asarray([r['momenta'] for r in specs]),
            orders=np.asarray([r['orders'] for r in specs],int),coefficients=coeff,
            structure_logs=np.empty(len(ids),complex),coefficient_precision_bounds=np.empty(len(ids)))
        for j,spec in enumerate(specs):
            kin=decode_identity(spec['identity']);p=np.asarray(spec['momenta'])
            old = donor.node(ids[j]) if donor is not None and ids[j] in donor.locations else None
            enough = old is not None and all(a<=b for a,b in zip(spec['orders'],old['orders']))
            if enough:
                c=old['coefficients'][tuple(slice(n+1) for n in spec['orders'])]
                error=float(old['coefficient_precision_bounds'])
            else:
                c,_,error=stable_c_recursion_coefficients(
                    {'necklace':'necklace','pair_ope':'pair','comb_ope':'comb'}[kin['channel']],
                    central_charge=25.,external_weights=kin['external_weights'],
                    internal_weights=tuple(1+p*p),orders=tuple(spec['orders']))
                if old is not None:
                    overlap=tuple(slice(min(a,b)+1) for a,b in zip(spec['orders'],old['orders']))
                    a,b=c[overlap],old['coefficients'][overlap]
                    if np.any(abs(a-b)>2e-13+2e-11*np.maximum(abs(a),abs(b))):
                        raise ArithmeticError('new tensor disagrees with its saved overlap')
            coeff[(j,*[slice(n+1) for n in spec['orders']])]=c
            reuse_structure=(old is not None and donor.plan['special_dps']>=plan['special_dps'])
            arrays['structure_logs'][j]=(old['structure_logs'] if reuse_structure
                                         else graph_structure_log(special,kin,p))
            arrays['coefficient_precision_bounds'][j]=error
            if error > 2e-11+2e-13: raise ArithmeticError('coefficient precision comparison failed')
            if progress: progress(dict(stage='built_node',chunk=index,node=j+1,nodes=len(ids)))
        if any(not np.all(np.isfinite(a)) for a in arrays.values()):
            raise FloatingPointError('nonfinite CFT chunk; no checkpoint published')
        folder.parent.mkdir(parents=True,exist_ok=True)
        temporary=Path(tempfile.mkdtemp(prefix='.partial-',dir=str(folder.parent)))
        try:
            with (temporary/'data.npz').open('xb') as f: np.savez_compressed(f,**arrays)
            write_json(temporary/'receipt.json',dict(format=CUSP_BANK_FORMAT,plan_sha256=plan_sha,
                node_ids=ids,data_sha256=digest(temporary/'data.npz'),orders=list(maximum),
                source_sha256=fingerprint(plan['source_files']),production_ready=False))
            os.rename(temporary,folder)
        except BaseException:
            import shutil
            shutil.rmtree(temporary,ignore_errors=True)
            raise
        if progress: progress(dict(stage='committed_chunk',chunk=index))


class CuspBank:
    """Verified finite tensor and one COMPLETE momentum rule, including dP."""
    def __init__(self, metadata, arrays):
        if metadata.get('format')!=CUSP_BANK_FORMAT or metadata.get('measure')!='product_dP':
            raise ValueError('new cusp bank with original measure required')
        self.metadata=metadata
        self.orders=tuple(integer(n,'block order') for n in metadata['orders'])
        self.arrays={k:np.asarray(v).copy() for k,v in arrays.items()}
        n=len(self.arrays['momenta'])
        shapes=dict(momenta=(n,3),log_weights=(n,),structure_logs=(n,),
                    coefficient_precision_bounds=(n,),coefficients=(n,*[o+1 for o in self.orders]))
        if not n or len(self.orders)!=3: raise ValueError('nonempty three-axis bank required')
        for key,shape in shapes.items():
            a=self.arrays[key]
            if a.shape!=shape or not np.all(np.isfinite(a)):
                raise ValueError('invalid bank array '+key)
            a.flags.writeable=False
        e=self.arrays['coefficient_precision_bounds']
        if np.any(e<0) or np.any(e>2e-11+2e-13) or np.any(self.arrays['momenta']<=0):
            raise ValueError('invalid verified spectral inputs')


class BankResolver:
    """Read-only, bounded chunk cache. Missing coefficients are errors, never zeros."""
    def __init__(self, root, *, cache_chunks=8):
        self.root=Path(root);self.plan=read_json(self.root/'plan.json');check_sources(self.plan)
        self.plan_sha=digest(self.root/'plan.json');self.chunks=chunk_nodes(self.plan)
        self.locations={key:(i,j) for i,ids in enumerate(self.chunks) for j,key in enumerate(ids)}
        self.cache=OrderedDict();self.cache_chunks=integer(cache_chunks,'cache size',1)

    def node(self, key):
        i,j=self.locations[key]
        if i not in self.cache:
            self.cache[i]=load_chunk(self.root,i,self.plan_sha,self.chunks[i])[1]
            while len(self.cache)>self.cache_chunks: self.cache.popitem(last=False)
        self.cache.move_to_end(i);data=self.cache[i]
        spec=self.plan['catalog']['nodes'][key]
        if (not np.array_equal(data['momenta'][j],spec['momenta']) or
                not np.array_equal(data['orders'][j],spec['orders'])):
            raise ValueError('catalog node data differs from the frozen request')
        return {name:value[j] for name,value in data.items()}

    def get(self, request_id):
        request=self.plan['catalog']['requests'][request_id]
        if fingerprint(request)!=request_id: raise ValueError('changed spectral request')
        orders=tuple(request['orders']);n=len(request['node_ids'])
        arrays=dict(momenta=np.empty((n,3)),log_weights=np.asarray(request['log_weights']),
            coefficients=np.empty((n,*[o+1 for o in orders]),complex),structure_logs=np.empty(n,complex),
            coefficient_precision_bounds=np.empty(n))
        for k,key in enumerate(request['node_ids']):
            data=self.node(key)
            spec=self.plan['catalog']['nodes'][key]
            if (spec['identity']!=request['identity'] or
                    any(a>b for a,b in zip(orders,spec['orders']))):
                raise ValueError('incompatible catalog coefficient tuple/order')
            for name in ('momenta','structure_logs','coefficient_precision_bounds'):
                arrays[name][k]=data[name]
            arrays['coefficients'][k]=data['coefficients'][tuple(slice(o+1) for o in orders)]
        return CuspBank(dict(format=CUSP_BANK_FORMAT,measure='product_dP',identity=request['identity'],
            orders=orders,source_manifest_sha256=fingerprint(self.plan['source_files']),
            request_id=request_id,production_ready=False),arrays)
