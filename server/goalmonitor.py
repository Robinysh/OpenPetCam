"""Read-only presentation of continuous goals, separate from work and completion.

The owner-authored catalog is synchronized by the existing metric producer.
Neither projecting a card nor reading this module dispatches work, rewrites
measurements, changes a goal, or creates a database/schema.
"""
import json
import math
from pathlib import Path
import sqlite3
import time


DEFAULT_FRESH_S = 90 * 60
QA_FRESH_S = 36 * 3600
TREND_POINTS = 288
TREND_DAYS = 7


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _rows(c, query, args=()):
    cursor = c.execute(query, args)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, tuple(row))) for row in cursor]


def _detail(raw):
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        value = raw
    observed = value.get('observed_at') if isinstance(value, dict) else None
    content = value.get('measurement', value) if isinstance(value, dict) else value
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False) if content is not None else ''
    return text, observed


def _health(goal, sample, now, catalog_known):
    max_age = QA_FRESH_S if goal == 'qa' else DEFAULT_FRESH_S

    def state(value, label, reason, expires=None):
        return {'state': value, 'label': label, 'reason': reason,
                'stale_after_s': max_age, 'expires_at': expires}

    if sample is None:
        return state('unmeasured', 'Not measured', 'No measurement is recorded for this goal.')
    if not catalog_known:
        return state('invalid', 'Definition unavailable', 'The current goal definition has not been synchronized.')
    stamp = sample['ts']
    if not _number(stamp) or stamp < 0 or stamp > now + 60:
        return state('invalid', 'Invalid timestamp', 'The latest record has no trustworthy measurement time.')
    if sample['target'] != sample['configured_target']:
        return state('invalid', 'Target changed', 'The recorded result used a different target; a new measurement is needed.')
    observed = sample['observed_at']
    if goal == 'qa' and not _number(observed):
        return state('invalid', 'Source time unknown', 'A cached QA result was recorded without its original observation time.')
    if goal in ('reports_answered', 'owner_reports') and observed is None:
        return state('invalid', 'Source not verified', 'Older report measurements could record zero after a failed collection; a verified source observation is needed.')
    stamp = observed if observed is not None else stamp
    if not _number(stamp) or stamp < 0 or stamp > now + 60:
        return state('invalid', 'Invalid source time', 'The source observation time is not trustworthy.')
    expires = stamp + max_age
    if now > expires:
        return state('stale', 'Stale', 'The last observation is older than this monitor’s freshness window.', expires)
    if not _number(sample['value']):
        return state('unmeasured', 'Not measured', 'The latest collection did not produce a numeric observation.', expires)
    if type(sample['ok']) is not bool:
        return state('invalid', 'Result unknown', 'The recorded measurement has no valid target assessment.', expires)
    return state('on_target' if sample['ok'] else 'off_target',
                 'On target' if sample['ok'] else 'Off target',
                 'Recorded assessment against the displayed target; the monitor remains open.', expires)


def project(c, now=None):
    """SELECT-only goal cards from an already-open SQLite connection."""
    now = time.time() if now is None else now
    if not _number(now):
        raise ValueError('now must be a finite timestamp')
    tables = {row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'metrics' not in tables:
        return {'goals': [], 'available': False, 'now': now, 'error': 'Goal measurements are not available yet.'}
    definitions = {}
    if 'goal_catalog' in tables:
        for row in _rows(c, 'SELECT id,definition,updated FROM goal_catalog ORDER BY id'):
            try:
                definition = json.loads(row['definition'])
                if not isinstance(definition, dict) or definition.get('id') != row['id']:
                    continue
            except (ValueError, TypeError):
                continue
            definitions[row['id']] = (definition, row['updated'])
    # Catalog membership is authoritative when populated. Historical samples for
    # retired goals remain in the database, but do not recreate retired cards.
    known = bool(definitions) or ('goal_catalog_state' in tables and c.execute('SELECT 1 FROM goal_catalog_state WHERE id=1').fetchone() is not None)
    identifiers = sorted(definitions) if known else [row[0] for row in c.execute('SELECT DISTINCT goal FROM metrics WHERE goal IS NOT NULL ORDER BY goal')]
    cards = []
    for ident in identifiers:
        definition, updated = definitions.get(ident, ({'id': ident, 'what': ident}, None))
        found = _rows(c, 'SELECT ts,value,target,ok,detail FROM metrics WHERE goal=? ORDER BY ts DESC,rowid DESC LIMIT 1', (ident,))
        raw = found[0] if found else None
        sample = None
        if raw is not None:
            detail, observed = _detail(raw['detail'])
            verdict = bool(raw['ok']) if type(raw['ok']) is int and raw['ok'] in (0, 1) else None
            sample = {'ts': raw['ts'], 'value': raw['value'] if _number(raw['value']) else None,
                      'target': str(raw['target'] or '').strip(), 'configured_target': str(definition.get('target', '')).strip(),
                      'ok': verdict, 'observed_at': observed, 'detail': detail,
                      'freshness_basis': 'source_observation' if observed is not None else 'collection_time'}
        trend = []
        for row in _rows(c, 'SELECT m.ts,m.value,m.ok FROM metrics m WHERE goal=? AND ts>=? AND ts<=? '
                             'AND rowid=(SELECT n.rowid FROM metrics n WHERE n.goal=m.goal AND n.ts=m.ts ORDER BY n.rowid DESC LIMIT 1) '
                             'ORDER BY ts DESC,rowid DESC LIMIT ?',
                             (ident, now - TREND_DAYS * 86400, now, TREND_POINTS)):
            trend.append([row['ts'], row['value'] if _number(row['value']) else None,
                          bool(row['ok']) if type(row['ok']) is int and row['ok'] in (0, 1) else None])
        trend.reverse()
        work = []
        if 'threads' in tables:
            found_thread = _rows(c, 'SELECT id,title,kind,status,worker,summary,created,updated FROM threads WHERE id=?', ('goal:' + ident,))
            thread = found_thread[0] if found_thread else None
            if thread is not None:
                thread = dict(thread)
                # An assignment or closed worker thread is not proof of a live
                # process or achieved goal. Keep those meanings separate.
                thread['worker_assigned'] = bool(thread.get('worker'))
                work.append(thread)
        health = _health(ident, sample, now, known)
        cards.append({'id': 'goal:' + ident, 'goal_id': ident, 'kind': 'goal', 'movable': False,
                      'title': definition.get('what') or ident, 'metric': definition.get('metric', ''),
                      'target': definition.get('target') or (sample['target'] if sample else ''),
                      'source': definition.get('ground_truth', ''), 'note': definition.get('note', ''),
                      'catalog_updated_at': updated,
                      'status': {'state': 'monitoring', 'label': 'Monitoring',
                                 'reason': 'Continuously monitored; meeting the target does not complete this goal.'},
                      'health': health, 'sample': sample, 'series': trend, 'work': work})
    return {'goals': cards, 'available': True, 'now': now, 'catalog_available': known}


def read(path, now=None):
    """Read an existing DB without opening a write-capable connection or migrating it."""
    stamp = time.time() if now is None else now
    try:
        uri = Path(path).resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(uri, uri=True, timeout=2) as c:
            c.row_factory = sqlite3.Row
            c.execute('PRAGMA query_only=ON')
            return project(c, now=stamp)
    except (OSError, sqlite3.Error):
        return {'goals': [], 'available': False, 'now': stamp,
                'error': 'Goal measurements are temporarily unavailable.'}
