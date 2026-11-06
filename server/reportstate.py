"""Report lifecycle from persisted evidence, separate from acknowledgement text.

An optional resolution has ``components``: nonempty objects with ``id``, ``title``,
``state`` and ``evidence``. Each completed component needs concrete evidence; a
legacy reply saying a fix is live is not verification of every requested behavior.
``monitoring`` keeps an outcome open for future observation. ``needs_review``
means a concrete question and its review material are ready for a person now.
Without a resolution, an answered ``reply.fix.review`` uses the same question /
evidence shape for a live outcome, no-change decision, or concrete plan.
"""
import time


STATES = {'done', 'in_progress', 'not_started', 'monitoring', 'needs_review', 'needs_hardware'}
LABELS = {'done': 'Done', 'in_progress': 'In progress', 'not_started': 'Backlog',
          'monitoring': 'Monitoring', 'needs_review': 'Needs review', 'needs_hardware': 'Needs hardware'}
LOOKING_TIMEOUT_S = 35 * 60


def _evidence(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_evidence(item) for item in value)
    if isinstance(value, dict):
        return any(_evidence(item) for item in value.values())
    return False


def review_ready(part):
    """A human can inspect this material and answer this question now."""
    review = part.get('review') if isinstance(part, dict) else None
    return (isinstance(review, dict) and isinstance(review.get('question'), str)
            and bool(review['question'].strip())
            and isinstance(review.get('evidence'), (str, list))
            and _evidence(review['evidence']))


def component_state(part):
    """Effective display state; legacy uncertainty cannot fill the review queue."""
    if not isinstance(part, dict) or not isinstance(part.get('state'), str) or part['state'] not in STATES:
        return 'not_started'
    state = part['state']
    if state == 'done' and not _evidence(part.get('evidence')):
        return 'not_started'
    if state == 'needs_review' and not review_ready(part):
        return 'not_started'
    return state


def classify(report, reply, resolution=None, now=None):
    """Return {state, label, reason}, with no IO or implicit completion claim."""
    def result(state, reason):
        return {'state': state, 'label': LABELS[state], 'reason': reason}

    if resolution is not None:
        parts = resolution.get('components') if isinstance(resolution, dict) else None
        if not isinstance(parts, list) or not parts:
            return result('not_started', 'The resolution has no component evidence; investigation is still needed.')
        states = []
        ids = set()
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get('state'), str) or part['state'] not in STATES:
                return result('not_started', 'A resolution component is malformed and needs investigation.')
            ident = part.get('id')
            if not isinstance(ident, str) or not ident.strip() or ident in ids:
                return result('not_started', 'Resolution components need unique identifiers before their outcomes can be verified.')
            ids.add(ident)
            state = part['state']
            if state == 'done' and not _evidence(part.get('evidence')):
                return result('not_started', 'A completed component has no verification evidence; verification work is still needed.')
            states.append(component_state(part))
        count = states.count('done')
        if count == len(states):
            return result('done', f'All {count} requested components have recorded verification evidence.')
        for state in ('needs_review', 'needs_hardware', 'in_progress', 'not_started', 'monitoring'):
            if state in states:
                if state == 'needs_review':
                    question = next(part['review']['question'].strip() for part in parts
                                    if component_state(part) == 'needs_review')
                    return result(state, question)
                if state == 'monitoring':
                    return result(state, 'Keep an eye on this in normal use; no immediate review or decision is requested.')
                return result(state, f'{count} of {len(states)} requested components are verified complete.')

    if reply is None:
        return result('not_started', 'No investigation outcome is recorded yet.')
    if not isinstance(reply, dict) or not reply:
        return result('not_started', 'The recorded reply is missing or malformed; investigation is still needed.')
    if reply.get('status') == 'looking':
        try:
            stamp = float(reply.get('ts') or (report.get('ts') if isinstance(report, dict) else None) or 0)
            age = (time.time() if now is None else now) - stamp
        except (ValueError, TypeError):
            age = float('inf')
        if 0 <= age < LOOKING_TIMEOUT_S:
            return result('in_progress', 'An investigation is running; completion is not yet recorded.')
        return result('not_started', 'The investigation has no outcome after its expected deadline; it needs another attempt.')
    fix = reply.get('fix')
    if not isinstance(fix, dict):
        return result('not_started', 'The reply has no recorded implementation outcome.')
    status = fix.get('status')
    if status == 'queued':
        return result('in_progress', 'A draft is queued; deployment and verification are pending.')
    if reply.get('status') == 'answered' and status in ('live', 'no_change', 'planned') and review_ready(fix):
        return result('needs_review', fix['review']['question'].strip())
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
    reason = reasons.get(status) if isinstance(status, str) else None
    return result('not_started', reason or 'No verified resolution is recorded for this request.')
