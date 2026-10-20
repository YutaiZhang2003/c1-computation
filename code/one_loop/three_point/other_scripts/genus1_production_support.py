"""Production configuration, checked records, and worker resource limits."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import os
import tempfile
import errno
import shutil
import signal
import socket
import stat
import time
import uuid
from datetime import datetime, timezone
import numpy as np

from genus1_boundary_cubature import fingerprint, TimeLimit, Controls


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    handle,name=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(handle,'w') as out:
            out.write(payload);out.flush();os.fsync(out.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


@contextmanager
def run_lock(directory):
    """Avoid two local processes spending the same diagnostic allocation."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'run.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('this local run is already active')
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


class Records:
    def __init__(self,identity,path=None):
        if not identity:raise ValueError('immutable record identity required')
        if path is not None:Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(':memory:' if path is None else str(path));self._transaction=False
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)')
        digest=fingerprint(identity);old=self.db.execute('SELECT identity FROM metadata').fetchall()
        if old and old!=[(digest,)]:
            self.db.close();raise ValueError('record identity changed')
        if not old:self.db.execute('INSERT INTO metadata VALUES (?)',(digest,))
        self.db.execute('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY,payload TEXT NOT NULL,sha TEXT NOT NULL)')
        self.db.commit()

    def get(self,key):
        row=self.db.execute('SELECT payload,sha FROM records WHERE key=?',(key,)).fetchone()
        if row is None:return None
        if hashlib.sha256(row[0].encode()).hexdigest()!=row[1]:raise ValueError('corrupt local record')
        return json.loads(row[0])

    def put(self,key,value):
        payload=json.dumps(value,sort_keys=True,allow_nan=False)
        self.db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)',
            (key,payload,hashlib.sha256(payload.encode()).hexdigest()))
        if not self._transaction:self.db.commit()

    @contextmanager
    def transaction(self):
        """Commit one bounded geometry batch; rollback an interrupted batch."""
        if self._transaction:raise RuntimeError('nested record transaction')
        self._transaction=True
        try:
            yield self
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        finally:self._transaction=False

    def close(self):self.db.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def utc(): return datetime.now(timezone.utc).isoformat()


def read(path): return json.loads(Path(path).read_text())


def checked(path):
    row = read(path)
    if fingerprint(row['payload']) != row['sha256']: raise ValueError('changed record: '+str(path))
    return row['payload']


def save(path, payload):
    write_json(path, dict(payload=payload, sha256=fingerprint(payload)))


def copy(source, destination):
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    expected = sha(source)
    if not destination.exists(): shutil.copy2(source, destination)
    if sha(destination) != expected: raise ValueError('changed deployment input')
    return expected

@contextmanager
def cpu_deadline(seconds,exception=TimeLimit):
    """POSIX process-CPU deadline, including an unusually slow native call.

    Python handles the signal at its next safe interpreter boundary. Any
    delayed delivery is measured and reported; no interrupted batch is used.
    """
    if seconds<=0:raise exception('process CPU allocation exhausted')
    if signal.getitimer(signal.ITIMER_PROF)!=(0.,0.):raise RuntimeError('nested CPU timer')
    previous=signal.getsignal(signal.SIGPROF)
    def expired(*_):raise exception('process CPU deadline reached')
    signal.signal(signal.SIGPROF,expired);signal.setitimer(signal.ITIMER_PROF,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,previous)


@contextmanager
def ownership(directory):
    """One live owner per tile; killed owners require explicit reconciliation."""
    directory=Path(directory);directory.parent.mkdir(parents=True,exist_ok=True)
    try:directory.mkdir()
    except FileExistsError:raise RuntimeError('tile has an owner; verify terminal ownership before recovery: '+str(directory))
    token=str(uuid.uuid4())
    save(directory/'owner.json',dict(token=token,pid=os.getpid(),host=socket.gethostname(),
        job_id=os.environ.get('SLURM_JOB_ID'),utc=utc()))
    try:yield
    finally:
        if checked(directory/'owner.json')['token']!=token:raise ValueError('claim owner changed')
        (directory/'owner.json').unlink()
        # NFS can briefly retain a silly-rename file after the last reader
        # closes owner.json. Never delete those files or any unknown member.
        # A killed owner still requires the explicit reconciliation above.
        for attempt in range(11):
            try:
                directory.rmdir()
                break
            except OSError as exc:
                if exc.errno!=errno.ENOTEMPTY:raise
                members=list(directory.iterdir())
                for path in members:
                    try:mode=path.lstat().st_mode
                    except FileNotFoundError:continue
                    if not path.name.startswith('.nfs') or not stat.S_ISREG(mode):raise
                if attempt==10:
                    # The verified owner has finished and owner.json is gone.
                    # Preserve lingering NFS evidence outside the active key;
                    # its eventual close must not block a future owner.
                    directory.rename(directory.with_name('.released-'+token))
                    break
                time.sleep(.05)

PRODUCTION_FORMAT = 'genus1-local-analytic-production-v1'

def template():
    return dict(format=PRODUCTION_FORMAT,run=dict(t=.25,q_max=.05,pair_radius=.125,triple_radius=.5,
        collision_height_split=8.,angle_order=16),compact_seed=20260921,
        providers=dict(minimum_momentum_nodes=8,minimum_elliptic_order=4,elliptic_orders=[2,4],banks=[],
            primary_regions=[],native_regions=[],height_banks={},
            coordinates=[dict(dps=45,phase_step=.2,maximum_nome=.95),
                         dict(dps=65,phase_step=.1,maximum_nome=.95)]),
        boundary_controls=dict(total_absolute_tolerance=4e-7,overrides={}),
        absolute_error_target=4.4e-6,
        diagnostics=dict(max_locations=48,max_parents=96,cpu_fraction=.1),preparations=[])


def make_plan(config):
    import genus1_boundary_integration as driver
    from genus1_analytic_boundary_layer import BoundaryPolicy
    if config.get('format')!=PRODUCTION_FORMAT:raise ValueError('unsupported local configuration')
    r=config['run'];policy=BoundaryPolicy(**{k:r[k] for k in ('q_max','pair_radius','triple_radius')})
    return driver.RunPlan(t=r['t'],policy=policy,collision_height_split=r['collision_height_split'],
        angle_order=r['angle_order'],mode='analytic')


def controls(config):
    import genus1_boundary_integration as driver
    c=config['boundary_controls'];tolerance=c['total_absolute_tolerance']
    ceilings={'primary':1500,'pair_face_analytic':700,'triple_pair_face':1500,
              'triple_scale_face':900,'triple_corner':400}
    result={}
    if set(c.get('overrides',{}))-set(driver.term_names('analytic')):raise ValueError('unknown component override')
    for name in driver.term_names('analytic'):
        kind,label=name.split(':');family=kind if kind=='primary' else label
        kw=dict(absolute_tolerance=tolerance/14,max_evaluations=ceilings[family])
        kw.update(c.get('overrides',{}).get(name,{}));result[name]=Controls(**kw)
    return result
