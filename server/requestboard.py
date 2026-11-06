"""Persistent request lanes, kept separate from verification and original replies.

A lane move records the owner's workflow intent against an evidence snapshot.
New evidence supersedes that intent. Reopened completion remains open across
reply-only updates until its resolution changes. Moving a card cannot invent
verification. The revision includes the last move, providing optimistic locking.
"""
import hashlib
import errno
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time

import reportstate


STATES = {'not_started', 'in_progress', 'monitoring', 'needs_review', 'blocked', 'done'}
LABELS = {'not_started': 'Backlog', 'in_progress': 'In progress',
          'monitoring': 'Monitoring', 'needs_review': 'Needs review', 'blocked': 'Blocked', 'done': 'Done'}
_LOCK = threading.RLock()
_ID = re.compile(r'\d{8}-\d{6}', re.ASCII)
FRAME_MAX_BYTES = 8 * 1024 * 1024


class Conflict(ValueError):
    """The request changed after the client fetched its revision (HTTP 409)."""


def _hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _basis(report, reply, resolution):
    # Observation/video snapshots can be large and are not request status inputs.
    identity = {key: report.get(key) for key in ('id', 'ts', 'by', 'note')}
    return _hash({'report': identity, 'reply': reply, 'resolution': resolution})


def decorate(report, reply, resolution, workflow=None):
    """Derive a lane without mutating the evidence or trusting a stale move."""
    if not isinstance(report, dict):
        raise ValueError('Request metadata must be an object.')
    try:
        verified = reportstate.classify(report, reply, resolution)
    except TypeError:
        # Historical JSON can contain an array/object where a component state
        # should be a string. Keep that request visible and fail completion shut.
        verified = {'state': 'not_started', 'label': 'Backlog',
                    'reason': 'The saved verification evidence is malformed and needs investigation.'}
    basis = _basis(report, reply, resolution)
    revision = _hash({'basis': basis, 'workflow': workflow})
    effective = dict(verified)
    active = None
    valid_workflow = (isinstance(workflow, dict) and isinstance(workflow.get('state'), str)
                      and workflow['state'] in STATES)
    if valid_workflow and workflow['state'] == 'needs_review':
        # A legacy review lane without a decision prompt is unfinished work,
        # not an actionable request for the owner's attention.
        valid_workflow = isinstance(workflow.get('reason'), str) and bool(workflow['reason'].strip())
    # A new "looking" reply is not fresh verification of a reopened regression.
    # Keep that reopening attached to its old resolution until evidence changes.
    reopened = (valid_workflow and workflow['state'] != 'done'
                and workflow.get('reopened_verified_done') is True
                and workflow.get('resolution_hash') == _hash(resolution))
    if valid_workflow and (workflow.get('basis') == basis or reopened):
        desired = workflow['state']
        # A corrupt or hand-edited workflow can never bypass the evidence gate.
        if desired != 'done' or verified['state'] == 'done':
            active = workflow
            reason = str(workflow.get('reason') or '').strip()
            if desired == 'done':
                effective = dict(verified)
            else:
                effective = {'state': desired, 'label': LABELS[desired],
                             'reason': reason or 'Moved on your request board; verification evidence is unchanged.'}
    return {'status': effective, 'verified_status': verified,
            'revision': revision, 'workflow': active}


def _directory(root, report_id):
    if not isinstance(report_id, str) or not _ID.fullmatch(report_id):
        raise ValueError('Invalid request identifier.')
    base = Path(root).resolve()
    directory = base / report_id
    if directory.resolve().parent != base:
        raise ValueError('Invalid request directory.')
    if not directory.is_dir():
        raise FileNotFoundError('Request not found.')
    return directory


def _read(directory, name, required=False):
    try:
        with (directory / (name + '.json')).open() as handle:
            return json.load(handle)
    except (OSError, ValueError):
        if required:
            raise FileNotFoundError('Request metadata is unavailable.') from None
        return None


def _frame_bytes(directory, full=False):
    """Open only a regular saved JPEG in this exact report directory.

    Descriptor-relative, no-follow opens keep a symlink replacement from
    escaping the evidence directory. Checking availability reads only its
    header; the HTTP read has the same size bound.
    """
    directory_fd = frame_fd = None
    try:
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        frame_fd = os.open('frame.jpg', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                           dir_fd=directory_fd)
        info = os.fstat(frame_fd)
        if not stat.S_ISREG(info.st_mode) or not 3 <= info.st_size <= FRAME_MAX_BYTES:
            raise FileNotFoundError('The saved report image is unavailable.')
        with os.fdopen(frame_fd, 'rb') as handle:
            frame_fd = None
            data = handle.read(FRAME_MAX_BYTES + 1 if full else 3)
        if not data.startswith(b'\xff\xd8\xff') or len(data) > FRAME_MAX_BYTES:
            raise FileNotFoundError('The saved report image is unavailable.')
        return data
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError('Invalid request image path.') from None
        raise FileNotFoundError('The saved report image is unavailable.') from None
    finally:
        if frame_fd is not None:
            os.close(frame_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def frame(root, report_id):
    """Read retained evidence, never a new camera snapshot."""
    with _LOCK:
        directory = _directory(root, report_id)
        report = _read(directory, 'report', required=True)
        if not isinstance(report, dict):
            raise FileNotFoundError('Request metadata is unavailable.')
        return _frame_bytes(directory, full=True)


def _load(root, report_id):
    directory = _directory(root, report_id)
    report = _read(directory, 'report', required=True)
    if not isinstance(report, dict):
        raise ValueError('Request metadata must be an object.')
    reply, resolution = _read(directory, 'reply'), _read(directory, 'resolution')
    workflow = _read(directory, 'workflow')
    decoration = decorate(report, reply, resolution, workflow)
    client = report.get('client')
    row = {'id': report_id, 'ts': report.get('ts'), 'by': report.get('by'),
           'note': report.get('note'),
           'shown': client.get('shown_status') if isinstance(client, dict) else None,
           'reply': reply, 'resolution': resolution, **decoration}
    try:
        _frame_bytes(directory)
    except (OSError, ValueError):
        pass
    else:
        row['evidence'] = {'frame': '/api/reports/frame?id=' + report_id}
    return directory, report, reply, resolution, workflow, row


def load(root, report_id):
    """Read one request row in the same shape used by the reports endpoint."""
    with _LOCK:
        return _load(root, report_id)[-1]


def move(root, report_id, state, revision, actor, reason=''):
    """Persist a lane move after validating evidence and the client snapshot.

    ``Conflict`` means the caller should refresh. Other ``ValueError`` instances
    are invalid moves. The only file written is the private ``workflow.json``.
    """
    if not isinstance(state, str) or state not in STATES:
        raise ValueError('Choose a valid request column.')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{64}', revision):
        raise ValueError('A request revision is required. Refresh the board.')
    if not isinstance(reason, str) or len(reason) > 1000:
        raise ValueError('The move note must be at most 1000 characters.')
    reason = reason.strip()
    if state == 'blocked' and not reason:
        raise ValueError('Add what is blocking this request.')
    if state == 'needs_review' and not reason:
        raise ValueError('Add what you can review or decide now.')
    if not isinstance(actor, str) or not actor.strip() or len(actor) > 200:
        raise ValueError('A signed-in device is required to move requests.')
    with _LOCK:
        directory, report, reply, resolution, old, row = _load(root, report_id)
        if row['revision'] != revision:
            raise Conflict('This request changed. Refresh it before moving it.')
        if state == 'done' and row['verified_status']['state'] != 'done':
            raise ValueError('Done requires verification evidence for every requested component. Review the checklist first.')
        history = old.get('history', []) if isinstance(old, dict) else []
        if not isinstance(history, list):
            history = []
        event = {'state': state, 'reason': reason, 'actor': actor.strip(),
                 'updated_at': time.time(), 'basis': _basis(report, reply, resolution),
                 'resolution_hash': _hash(resolution),
                 'reopened_verified_done': row['verified_status']['state'] == 'done' and state != 'done'}
        workflow = {**event, 'history': (history + [event])[-100:]}
        data = json.dumps(workflow, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                             prefix='.workflow-', delete=False) as handle:
                temporary = handle.name
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, directory / 'workflow.json')
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        # If another process published evidence during this move, the stored
        # basis is already stale and _load correctly returns the new evidence.
        return _load(root, report_id)[-1]
