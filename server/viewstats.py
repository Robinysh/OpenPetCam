"""Conservative saved-view statistics from existing Follow logs; no camera actions.

Rows are mappings (including sqlite3.Row). Fetch track context at least 33 seconds
before and after the observation range. These are logged detector observations,
not ground-truth visits or full-day coverage; cached detection boxes can repeat.
"""
from bisect import bisect_right
from collections import deque
from itertools import pairwise
import json
import math


MAX_POSE_AGE_S = 30.0
SETTLE_S = 3.0
MAX_SAMPLE_S = 6.0
VISIBLE_SCORE = 0.15                 # watch.py ABSENT_SCORE / accumulate(visible)
POSE_TOL_STEPS = 2.0
SUGGESTION_MIN_VISIBLE_S = 30.0


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def _box(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return False
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return False
    xy = [_number(x) for x in value]
    return all(x is not None for x in xy) and 0 <= xy[0] < xy[2] <= 1 and 0 <= xy[1] < xy[3] <= 1


def _same(a, b):
    return a is not None and b is not None and all(abs(x - y) <= POSE_TOL_STEPS for x, y in zip(a, b))


def _overlap(a, b, frame):
    return max(0.0, 1 - abs(a[0] - b[0]) / frame[0]) * max(0.0, 1 - abs(a[1] - b[1]) / frame[1])


def _merge_windows(windows):
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _stationary(track_rows):
    groups = {}
    dropped = 0
    for raw in track_rows:
        row = dict(raw)
        stamp = _number(row.get('ts'))
        if stamp is None:
            dropped += 1
            continue
        pan, tilt = _number(row.get('pan')), _number(row.get('tilt'))
        pose = (pan, tilt) if pan is not None and tilt is not None and min(pan, tilt) >= 0 else None
        moves = _number(row.get('moves', 0) or 0)
        idle = row.get('act') in (None, '', 'idle') and moves == 0 and not row.get('motor_moving')
        groups.setdefault(stamp, []).append((pose, idle))
    tracks, blocked = [], []
    for stamp, items in sorted(groups.items()):
        pose, idle = items[0]
        if any(other != items[0] for other in items[1:]):
            pose, idle = None, False
        tracks.append((stamp, pose, idle))
        if not idle or pose is None:
            # A logged action can precede the next pose-cache refresh by 30 seconds.
            blocked.append((stamp, stamp + MAX_POSE_AGE_S + SETTLE_S))
    if tracks:
        # Without context outside the query, a cached turn just before/after the
        # supplied track range cannot be excluded. Trim those uncertain edges.
        blocked.extend([(tracks[0][0], tracks[0][0] + MAX_POSE_AGE_S + SETTLE_S),
                        (tracks[-1][0] - MAX_POSE_AGE_S, tracks[-1][0])])
    for a, b in zip(tracks, tracks[1:]):
        if a[1] is not None and b[1] is not None and not _same(a[1], b[1]):
            # A newly observed pose can describe a turn that happened up to 30 s ago.
            blocked.append((b[0] - MAX_POSE_AGE_S, b[0] + SETTLE_S))
    blocked = _merge_windows(blocked)
    ends = [window[1] for window in blocked]
    segments = []
    for a, b in zip(tracks, tracks[1:]):
        if not (a[2] and b[2] and _same(a[1], b[1]) and 0 < b[0] - a[0] <= MAX_POSE_AGE_S):
            continue
        cursor = a[0]
        k = bisect_right(ends, cursor)
        while k < len(blocked) and blocked[k][0] < b[0]:
            lo, hi = blocked[k]
            if lo > cursor:
                segments.append((cursor, min(lo, b[0]), a[1]))
            cursor = max(cursor, hi)
            k += 1
        if cursor < b[0]:
            segments.append((cursor, b[0], a[1]))
    return segments, dropped, len(tracks)


def _new_stats(**fields):
    return dict(fields, observed_seconds=0.0, visible_seconds=0.0, visits=0, observed_visits=0,
                _last_end=None, _visible_end=None, _pan_sum=0.0, _tilt_sum=0.0)


def _add(stats, start, end, visible, pose):
    duration = end - start
    if stats['_last_end'] is None or start > stats['_last_end'] + 1e-6:
        stats['observed_visits'] += 1
    if visible and (stats['_visible_end'] is None or start > stats['_visible_end'] + 1e-6):
        stats['visits'] += 1
    stats['_last_end'] = end
    stats['_visible_end'] = end if visible else None
    stats['observed_seconds'] += duration
    stats['visible_seconds'] += duration if visible else 0
    stats['_pan_sum'] += duration * pose[0]
    stats['_tilt_sum'] += duration * pose[1]


def _public(stats):
    return {key: round(value, 3) if isinstance(value, float) else value
            for key, value in stats.items() if not key.startswith('_')}


def _settings(presets, pan_frame, tilt_frame):
    frame = (_number(pan_frame), _number(tilt_frame))
    if any(v is None or v <= 0 for v in frame):
        raise ValueError('frame dimensions must be finite positive motor-step spans')
    views = []
    for raw in presets:
        row = dict(raw)
        pan, tilt = _number(row.get('x')), _number(row.get('y'))
        if not isinstance(row.get('name'), str) or not row['name'].strip() or pan is None or tilt is None \
                or min(pan, tilt) < 0:
            raise ValueError('each preset needs a name and finite x/y coordinates')
        views.append(_new_stats(name=row['name'], x=pan, y=tilt))
    return frame, views


def summarize(obs_rows, track_rows, presets, pan_frame=733, tilt_frame=424):
    """Accept unordered rows; production SQLite cursors should use summarize_sorted."""
    frame, views = _settings(presets, pan_frame, tilt_frame)
    segments, invalid_track, track_count = _stationary(track_rows)
    groups, invalid_obs = {}, 0
    for raw in obs_rows:
        row = dict(raw)
        stamp, score = _number(row.get('ts')), _number(row.get('score'))
        if stamp is None:
            invalid_obs += 1
            continue
        visible = score is not None and score >= VISIBLE_SCORE and _box(row.get('box'))
        valid = score is not None and 0 <= score <= 1 and (score < VISIBLE_SCORE or _box(row.get('box')))
        groups.setdefault(stamp, []).append((valid, visible))
    obs = [(stamp, all(v[0] for v in rows), all(v[1] for v in rows)) for stamp, rows in sorted(groups.items())]
    invalid_obs += sum(not valid for _, valid, _ in obs)
    counts = {'observation_rows': len(obs), 'track_rows': track_count, 'invalid_observations': invalid_obs,
              'invalid_track_rows': invalid_track, 'first_observation_ts': obs[0][0] if obs else None,
              'last_observation_ts': obs[-1][0] if obs else None}
    return _summarize_points(iter(obs), iter(segments), views, frame, counts)


def _ordered_points(rows, kind, counts):
    """Fold same-time duplicates in constant memory, rejecting out-of-order cursors."""
    previous, combined = None, None

    def emit(stamp, value):
        counts['track_rows' if kind == 'track' else 'observation_rows'] += 1
        if kind == 'obs':
            counts['invalid_observations'] += not value[0]
            if counts['first_observation_ts'] is None:
                counts['first_observation_ts'] = stamp
            counts['last_observation_ts'] = stamp
        return stamp, *value

    for raw in rows:
        row = dict(raw)
        stamp = _number(row.get('ts'))
        if stamp is None:
            counts['invalid_track_rows' if kind == 'track' else 'invalid_observations'] += 1
            continue
        if previous is not None and stamp < previous:
            raise ValueError(f'{kind} rows must be ordered by ts ascending')
        if kind == 'track':
            pan, tilt = _number(row.get('pan')), _number(row.get('tilt'))
            pose = (pan, tilt) if pan is not None and tilt is not None and min(pan, tilt) >= 0 else None
            moves = _number(row.get('moves', 0) or 0)
            idle = row.get('act') in (None, '', 'idle') and moves == 0 and not row.get('motor_moving')
            value = pose, idle
        else:
            score = _number(row.get('score'))
            visible = score is not None and score >= VISIBLE_SCORE and _box(row.get('box'))
            valid = score is not None and 0 <= score <= 1 and (score < VISIBLE_SCORE or _box(row.get('box')))
            value = valid, visible
        if previous == stamp:
            combined = ((combined[0] and value[0], combined[1] and value[1]) if kind == 'obs'
                        else combined if combined == value else (None, False))
        else:
            if previous is not None:
                yield emit(previous, combined)
            previous, combined = stamp, value
    if previous is not None:
        yield emit(previous, combined)


def _stationary_stream(points):
    """Keep only the track lookahead needed to exclude late cached pose changes."""
    source, buffered, blocked = iter(points), deque(), []
    previous, finished = None, False

    def read_one():
        nonlocal previous, finished
        try:
            point = next(source)
        except StopIteration:
            finished = True
            if previous:
                blocked.append((previous[0] - MAX_POSE_AGE_S, previous[0]))
            return
        stamp, pose, idle = point
        if previous is None:
            blocked.append((stamp, stamp + MAX_POSE_AGE_S + SETTLE_S))
        elif previous[1] is not None and pose is not None and not _same(previous[1], pose):
            blocked.append((stamp - MAX_POSE_AGE_S, stamp + SETTLE_S))
        if not idle or pose is None:
            blocked.append((stamp, stamp + MAX_POSE_AGE_S + SETTLE_S))
        buffered.append(point)
        previous = point

    while len(buffered) < 2 and not finished:
        read_one()
    while len(buffered) >= 2:
        a, b = buffered[0], buffered[1]
        while not finished and buffered[-1][0] < b[0] + MAX_POSE_AGE_S:
            read_one()
        blocked[:] = _merge_windows(window for window in blocked if window[1] > a[0])
        if a[2] and b[2] and _same(a[1], b[1]) and 0 < b[0] - a[0] <= MAX_POSE_AGE_S:
            cursor = a[0]
            for lo, hi in blocked:
                if lo >= b[0]:
                    break
                if lo > cursor:
                    yield cursor, min(lo, b[0]), a[1]
                cursor = max(cursor, hi)
            if cursor < b[0]:
                yield cursor, b[0], a[1]
        buffered.popleft()
        while len(buffered) < 2 and not finished:
            read_one()


def summarize_sorted(obs_rows, track_rows, presets, pan_frame=733, tilt_frame=424):
    """One pass over ts-ascending SQLite cursors with approximately 30 s lookahead.

    Do not fetchall() first. Provide track context at least 33 s around the obs
    range. Duplicate timestamps, partial intervals, cache uncertainty and visits
    have the same semantics as summarize(); neither iterator is materialized.
    """
    frame, views = _settings(presets, pan_frame, tilt_frame)
    counts = {'observation_rows': 0, 'track_rows': 0, 'invalid_observations': 0,
              'invalid_track_rows': 0, 'first_observation_ts': None, 'last_observation_ts': None}
    obs = _ordered_points(obs_rows, 'obs', counts)
    segments = _stationary_stream(_ordered_points(track_rows, 'track', counts))
    return _summarize_points(obs, segments, views, frame, counts)


def _summarize_points(obs, segments, views, frame, counts):
    total, bins = _new_stats(), {}
    eligible_seconds, contributing = 0.0, 0
    segment = next(segments, None)
    for (start, valid, visible), (next_t, _, _) in pairwise(obs):
        end = min(start + MAX_SAMPLE_S, next_t)
        if not valid or end <= start:
            continue
        eligible_seconds += end - start
        while segment is not None and segment[1] <= start:
            segment = next(segments, None)
        contributed = False
        while segment is not None and segment[0] < end:
            lo, hi, pose = segment
            lo, hi = max(start, lo), min(end, hi)
            if hi > lo:
                overlaps = [_overlap(pose, (view['x'], view['y']), frame) for view in views]
                best = max(range(len(views)), key=lambda k: overlaps[k]) if views else None
                if best is not None and overlaps[best] > 0.5:
                    target = views[best]
                else:
                    key = tuple(math.floor(p / (span / 4) + 0.5) for p, span in zip(pose, frame))
                    target = bins.setdefault(key, _new_stats())
                _add(target, lo, hi, visible, pose)
                _add(total, lo, hi, visible, pose)
                contributed = True
            if segment[1] > end:
                break
            segment = next(segments, None)
        contributing += contributed
    # Finish counting the bounded query context; this does not retain its rows.
    for _ in segments:
        pass
    candidates = []
    for stats in sorted(bins.values(), key=lambda v: (-v['visible_seconds'], -v['observed_seconds'], v['_pan_sum'])):
        if stats['visible_seconds'] < SUGGESTION_MIN_VISIBLE_S:
            continue
        pose = (stats['_pan_sum'] / stats['observed_seconds'], stats['_tilt_sum'] / stats['observed_seconds'])
        if any(_overlap(pose, (v['x'], v['y']), frame) > 0.5 for v in views + candidates):
            continue
        candidates.append(dict(_public(stats), x=round(pose[0], 1), y=round(pose[1], 1)))
        if len(candidates) == 5:
            break
    return {
        'scope': 'Follow-logged stationary intervals',
        'observed_seconds': round(total['observed_seconds'], 3),
        'visible_seconds': round(total['visible_seconds'], 3),
        'views': [_public(view) for view in views], 'suggestions': candidates,
        'unsaved_observed_seconds': round(sum(v['observed_seconds'] for v in bins.values()), 3),
        'unsaved_visible_seconds': round(sum(v['visible_seconds'] for v in bins.values()), 3),
        'coverage': dict(counts, contributing_observations=contributing,
                         bounded_observation_seconds=round(eligible_seconds, 3),
                         excluded_seconds=round(eligible_seconds - total['observed_seconds'], 3)),
        'assumptions': {'maximum_sample_seconds': MAX_SAMPLE_S, 'maximum_pose_gap_seconds': MAX_POSE_AGE_S,
                        'pose_cache_seconds': MAX_POSE_AGE_S, 'settle_seconds': SETTLE_S,
                        'visible_score_threshold': VISIBLE_SCORE, 'minimum_view_overlap': 0.5,
                        'suggestion_min_visible_seconds': SUGGESTION_MIN_VISIBLE_S,
                        'visits': 'Contiguous logged-visible bouts; gaps start a new bout.',
                        'coverage': 'Only Follow-logged time with stationary pose evidence; not full-day pet presence.',
                        'detections': 'Scores and boxes can be cached; these are detector records, not verified pet visits.'},
    }
