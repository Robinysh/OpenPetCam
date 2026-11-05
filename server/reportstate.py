"""Report lifecycle from persisted evidence, separate from acknowledgement text.

An optional resolution has ``components``: nonempty objects with ``id``, ``title``,
``state`` and ``evidence``. Each completed component needs concrete evidence; a
legacy reply saying a fix is live is not verification of every requested behavior.
"""
import time


STATES = {'done', 'in_progress', 'not_started', 'needs_review', 'needs_hardware'}
LABELS = {'done': 'Done', 'in_progress': 'In progress', 'not_started': 'Not started',
          'needs_review': 'Needs review', 'needs_hardware': 'Needs hardware'}
LOOKING_TIMEOUT_S = 35 * 60


def _evidence(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_evidence(item) for item in value)
    if isinstance(value, dict):
        return any(_evidence(item) for item in value.values())
    return False


def classify(report, reply, resolution=None, now=None):
    """Return {state, label, reason}, with no IO or implicit completion claim."""
    def result(state, reason):
        return {'state': state, 'label': LABELS[state], 'reason': reason}

    if resolution is not None:
        parts = resolution.get('components') if isinstance(resolution, dict) else None
        if not isinstance(parts, list) or not parts:
            return result('needs_review', 'The resolution has no component evidence.')
        states = []
        ids = set()
        for part in parts:
            if not isinstance(part, dict) or part.get('state') not in STATES:
                return result('needs_review', 'A resolution component has an invalid state.')
            ident = part.get('id')
            if not isinstance(ident, str) or not ident.strip() or ident in ids:
                return result('needs_review', 'Resolution components need unique identifiers.')
            ids.add(ident)
            state = part['state']
            if state == 'done' and not _evidence(part.get('evidence')):
                return result('needs_review', 'A completed component has no verification evidence.')
            states.append(state)
        count = states.count('done')
        if count == len(states):
            return result('done', f'All {count} requested components have recorded verification evidence.')
        for state in ('needs_review', 'needs_hardware', 'in_progress', 'not_started'):
            if state in states:
                return result(state, f'{count} of {len(states)} requested components are verified complete.')

    if reply is None:
        return result('not_started', 'No investigation outcome is recorded yet.')
    if not isinstance(reply, dict) or not reply:
        return result('needs_review', 'The recorded reply is missing or malformed.')
    if reply.get('status') == 'looking':
        try:
            stamp = float(reply.get('ts') or (report.get('ts') if isinstance(report, dict) else None) or 0)
            age = (time.time() if now is None else now) - stamp
        except (ValueError, TypeError):
            age = float('inf')
        if 0 <= age < LOOKING_TIMEOUT_S:
            return result('in_progress', 'An investigation is running; completion is not yet recorded.')
        return result('needs_review', 'The investigation has no outcome after its expected deadline.')
    fix = reply.get('fix')
    if not isinstance(fix, dict):
        return result('needs_review', 'The reply has no recorded implementation outcome.')
    status = fix.get('status')
    if status == 'queued':
        return result('in_progress', 'A draft is queued; deployment and verification are pending.')
    reasons = {
        'live': 'A previous reply reports a live fix; all requested behavior still needs verification.',
        'no_change': 'An investigation replied without a verified implementation outcome.',
        'failed_checks': 'The draft failed its checks and was not verified complete.',
        'replay_worse': 'The draft failed the comparison against existing behavior.',
        'rolled_back': 'The attempted fix was rolled back.',
        'planned': 'The reply describes planned work without completion evidence.',
        'duplicate': 'The linked request and every requested component still need verification.',
        'redo': 'The previous draft no longer applied and needs another attempt.',
        'error': 'The investigation failed before recording a verified outcome.',
    }
    return result('needs_review', reasons.get(status, 'No verified resolution is recorded for this request.'))
