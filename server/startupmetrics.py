"""Page-load startup statistics; reconnect timing is a separate diagnostic.

The video element's ``playing`` event is the source observation. These
percentiles describe successful page loads, not presentation timestamps,
foreground resumes, or the success rate of all startup attempts.
"""
import json
import math
import time


WINDOW_DAYS = 7
MIN_SAMPLES = 5
TARGET = 'p50 <= 1, p90 <= 2'


def _number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _get(row, key):
    try:
        return row[key]
    except (IndexError, KeyError, TypeError):
        return None


def _percentile(values, fraction):
    # Preserve the existing collector's upper-order statistic; assess before
    # display rounding so a just-over-target observation cannot become green.
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] / 1000


def summarize(rows, now=None):
    """Summarize dictionary-like usage rows without IO, labels or inferred successes."""
    now = time.time() if now is None else now
    if not _number(now) or now < 0:
        raise ValueError('now must be a finite nonnegative timestamp')
    opens, connections, stamps = [], [], []
    timeouts = reconnects = 0
    for row in rows:
        stamp, name = _get(row, 'ts'), _get(row, 'name')
        if not _number(stamp) or stamp < 0 or not now - WINDOW_DAYS * 86400 < stamp <= now:
            continue
        if name == 'live_start_timeout':
            timeouts += 1
        elif name == 'live_reconnect':
            reconnects += 1
        elif name == 'live_first_frame':
            connection = _get(row, 'ms')
            if _number(connection) and connection >= 0:
                connections.append(connection)
            raw = _get(row, 'detail')
            try:
                detail = json.loads(raw) if isinstance(raw, str) else raw
            except (ValueError, RecursionError):
                detail = None
            opening = detail.get('open_ms') if isinstance(detail, dict) else None
            if _number(opening) and opening >= 0:
                opens.append(opening)
                stamps.append(stamp)
    return {'from': 'open', 'window_days': WINDOW_DAYS, 'min_samples': MIN_SAMPLES,
            'ready': len(opens) >= MIN_SAMPLES, 'n': len(opens),
            'p50': _percentile(opens, .5), 'p90': _percentile(opens, .9),
            'observed_at': max(stamps) if stamps else None,
            'first_observed_at': min(stamps) if stamps else None,
            'connect_n': len(connections), 'connect_p50': _percentile(connections, .5),
            'connect_p90': _percentile(connections, .9), 'timeouts': timeouts, 'reconnects': reconnects}


def goal_metric(stats=None):
    """Always return a goal sample, including an explicit unknown after failed collection."""
    supplied = stats if isinstance(stats, dict) else {}
    verified_source = (supplied.get('from') == 'open' and supplied.get('window_days') == WINDOW_DAYS
                       and supplied.get('min_samples') == MIN_SAMPLES)
    clean = {'from': 'open' if verified_source else 'unknown',
             'window_days': WINDOW_DAYS, 'min_samples': MIN_SAMPLES}
    for key in ('n', 'connect_n', 'timeouts', 'reconnects'):
        value = supplied.get(key)
        clean[key] = value if verified_source and type(value) is int and value >= 0 else None
    for key in ('p50', 'p90', 'observed_at', 'first_observed_at', 'connect_p50', 'connect_p90'):
        value = supplied.get(key)
        clean[key] = value if verified_source and _number(value) and value >= 0 else None
    ready = (supplied.get('ready') is True and clean['from'] == 'open'
             and supplied.get('window_days') == WINDOW_DAYS and supplied.get('min_samples') == MIN_SAMPLES
             and type(clean['n']) is int and clean['n'] >= MIN_SAMPLES and clean['observed_at'] is not None
             and clean['p50'] is not None and clean['p90'] is not None and clean['p90'] >= clean['p50'])
    clean['ready'] = ready

    def seconds(value):
        return f'{value:.3f} s' if value is not None else 'unavailable'

    def count(value):
        return str(value) if value is not None else 'unavailable'

    first, last = clean['first_observed_at'], clean['observed_at']
    span = f'; observations span {(last - first) / 3600:.1f} h' if first is not None and last is not None and first <= last else ''
    summary = (f"Page load -> playing: p50 {seconds(clean['p50'])}, p90 {seconds(clean['p90'])}; "
               f"{count(clean['n'])} successful page-open samples in 7 days (need {MIN_SAMPLES} to assess){span}. "
               f"Connection-only diagnostic: p50 {seconds(clean['connect_p50'])}, p90 {seconds(clean['connect_p90'])}, "
               f"{count(clean['connect_n'])} samples; never used to satisfy this goal. "
               f"{count(clean['timeouts'])} startup-timeout events and {count(clean['reconnects'])} reconnect events. "
               "Unsuccessful starts have no percentile sample; foreground-resume latency is not separately measured. "
               "Event counts are not an attempt success rate.")
    if not verified_source:
        summary = ('Startup measurement source unavailable; page-open, connection, timeout and reconnect counts '
                   'are unknown. No earlier success or connection-only timing is reused.')
    sample = {'goal': 'video_start', 'value': clean['p50'] if ready else None, 'target': TARGET,
              'ok': ready and clean['p50'] <= 1 and clean['p90'] <= 2,
              'detail': {'schema': 'startup-v1', 'summary': summary, 'stats': clean}}
    if clean['observed_at'] is not None:
        sample['observed_at'] = clean['observed_at']
    return sample
