"""Conservative, offline memory of learned visual obstructions at camera poses.

An obstruction probability must come from a visual model. A missed pet detection
is never evidence of an obstruction. Exclusion is temporary, requires repeated
independent high-quality observations, and always yields to scheduled probes.
No camera, training-label, network, or implicit filesystem access is provided.
Pose coordinates are exact keys in one explicitly fingerprinted setup.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


VERSION = 1
STATES = ('unknown', 'suspected', 'temporarily_excluded', 'reinspection_due')


def _number(value, name, minimum=0.0, maximum=None):
    if isinstance(value, bool):
        raise ValueError(name + ' must be a finite number')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(name + ' must be a finite number') from None
    if not math.isfinite(result) or result < minimum or maximum is not None and result > maximum:
        raise ValueError(name + ' is out of range')
    return result


def _pose(value):
    try:
        values = tuple(value)
    except TypeError:
        raise ValueError('pose must contain two finite coordinates') from None
    if len(values) != 2:
        raise ValueError('pose must contain two finite coordinates')
    return tuple(_number(v, 'pose coordinate', minimum=-float('inf')) for v in values)


def _context(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('context must be a nonempty fingerprint string')
    return value


@dataclass(frozen=True)
class ObstructionConfig:
    high_threshold: float = 0.9
    low_threshold: float = 0.2
    min_quality: float = 0.6
    max_frame_age_s: float = 2.5
    min_gap_s: float = 1.0
    min_samples: int = 3
    min_span_s: float = 10.0
    evidence_window_s: float = 120.0
    exclusion_ttl_s: float = 300.0
    probe_interval_s: float = 60.0
    max_poses: int = 512

    def __post_init__(self):
        for name in ('high_threshold', 'low_threshold', 'min_quality'):
            object.__setattr__(self, name, _number(getattr(self, name), name, maximum=1.0))
        if self.low_threshold >= self.high_threshold:
            raise ValueError('low_threshold must be less than high_threshold')
        for name in ('max_frame_age_s', 'min_gap_s', 'min_span_s',
                     'evidence_window_s', 'exclusion_ttl_s', 'probe_interval_s'):
            value = _number(getattr(self, name), name)
            if value <= 0:
                raise ValueError(name + ' must be positive')
            object.__setattr__(self, name, value)
        if type(self.min_samples) is not int or not 2 <= self.min_samples <= 256:
            raise ValueError('min_samples must be an integer in 2..256')
        if type(self.max_poses) is not int or self.max_poses <= 0:
            raise ValueError('max_poses must be a positive integer')
        if self.evidence_window_s < max(self.min_span_s,
                                        (self.min_samples - 1) * self.min_gap_s):
            raise ValueError('evidence_window_s cannot hold the required independent evidence')


# Short constructor name for callers building small experiments.
Config = ObstructionConfig


@dataclass(frozen=True)
class Observation:
    pose: object
    captured_at: float
    observed_at: float
    context: str
    settled: bool = True
    image_valid: bool = True
    quality: float = 1.0
    obstruction_probability: object = None
    pet_detected: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'pose', _pose(self.pose))
        object.__setattr__(self, 'context', _context(self.context))
        for name in ('captured_at', 'observed_at'):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        for name in ('settled', 'image_valid', 'pet_detected'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + ' must be a boolean')
        object.__setattr__(self, 'quality', _number(self.quality, 'quality', maximum=1.0))
        if self.obstruction_probability is not None:
            object.__setattr__(self, 'obstruction_probability',
                               _number(self.obstruction_probability, 'obstruction_probability', maximum=1.0))


class ObstructionMemory:
    """Fail-open view memory. Queries take monotonic time and current context.

    decision's context is authoritative. A changed mount/map/model fingerprint
    clears all beliefs. observe never changes an established context, so delayed
    old-context frames cannot roll it back. setup_changed also handles scene
    changes that do not alter the fingerprint (e.g. moving furniture).

    due is a read-only budget proposal: scheduling does not count as a probe.
    Only a fresh, settled, valid visual observation can refresh an inspection.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else ObstructionConfig()
        if not isinstance(self.config, ObstructionConfig):
            raise ValueError('config must be an ObstructionConfig')
        self._context = None
        self._context_since = 0.0
        self._now = 0.0
        self._records = {}
        self._probe_attempts = {}

    @property
    def context(self):
        return self._context

    def _time(self, now):
        now = _number(now, 'now')
        if now < self._now:
            raise ValueError('query time must not move backwards')
        self._now = now
        return now

    def setup_changed(self, context, now=None):
        """Invalidate every pose, even when the caller reuses the fingerprint."""
        context = _context(context)
        now = self._time(self._now if now is None else now)
        self._context, self._context_since = context, now
        self._records.clear()
        self._probe_attempts.clear()

    def _activate(self, context, now):
        context = _context(context)
        if self._context != context:
            self.setup_changed(context, now)

    def _result(self, pose, now):
        record = self._records.get(pose)
        state, skip, due = 'unknown', False, True
        reason, deadline, expiry, count = 'uninspected_view', self._context_since, None, 0
        if record is not None:
            times = [t for t in record['high_captures'] if now - t <= self.config.evidence_window_s]
            count = len(times)
            expiry = record['excluded_until']
            deadline = record['next_probe_at']
            due = now >= deadline
            reason = record['reason']
            if expiry is not None:
                deadline = min(deadline, expiry)
                due = now >= deadline
                if due:
                    state, reason = 'reinspection_due', 'exclusion_expired' if now >= expiry else 'mandatory_probe_due'
                else:
                    state, skip, reason = 'temporarily_excluded', True, 'repeated_independent_visual_obstruction'
            elif count:
                state, reason = 'suspected', 'insufficient_independent_visual_evidence'
            elif record['high_captures']:
                reason = 'obstruction_evidence_expired'
        return {'pose': list(pose), 'context': self._context, 'state': state,
                'skip': skip, 'probe_due': due, 'reason': reason,
                'probe_deadline': deadline, 'excluded_until': expiry,
                'high_observations': count,
                'last_probe_attempt_at': self._probe_attempts.get(pose, {}).get('observed_at'),
                'last_valid_observed_at': record['last_observed_at'] if record else None}

    def decision(self, pose, now, context):
        pose, context = _pose(pose), _context(context)
        now = self._time(now)
        self._activate(context, now)
        return self._result(pose, now)

    def observe(self, observation):
        if not isinstance(observation, Observation):
            raise ValueError('observe expects an Observation')
        o, c = observation, self.config
        # Observation contexts are never authoritative after initialization.
        if self._context is None:
            self.setup_changed(o.context, self._now)
        record = self._records.get(o.pose)
        attempt = self._probe_attempts.get(o.pose)
        latest_capture = max(record['last_captured_at'] if record else -1,
                             attempt['captured_at'] if attempt else -1)
        ignored = None
        if o.context != self._context:
            ignored = 'context_mismatch'
        elif o.observed_at < self._now:
            ignored = 'out_of_order_observation'
        elif o.captured_at < self._context_since:
            ignored = 'frame_predates_setup'
        elif o.captured_at > o.observed_at or o.observed_at - o.captured_at > c.max_frame_age_s:
            ignored = 'stale_or_future_frame'
        elif o.captured_at <= latest_capture:
            ignored = 'duplicate_or_older_frame'
        if ignored is None:
            # A real fresh probe attempt rotates scheduling even if its image
            # is unusable. It cannot extend any belief or inspection deadline.
            if o.pose not in self._probe_attempts and len(self._probe_attempts) >= c.max_poses:
                oldest = min(self._probe_attempts, key=lambda p: (self._probe_attempts[p]['observed_at'], p))
                del self._probe_attempts[oldest]
            self._probe_attempts[o.pose] = {'captured_at': o.captured_at, 'observed_at': o.observed_at}
            if not o.settled:
                ignored = 'camera_not_settled'
            elif not o.image_valid or o.quality < c.min_quality:
                ignored = 'invalid_or_low_quality_image'
            elif o.obstruction_probability is None and not o.pet_detected:
                ignored = 'no_visual_obstruction_evidence'
        if ignored is None and not o.pet_detected and o.obstruction_probability >= c.high_threshold and record:
            captures = record['high_captures']
            if captures and o.captured_at - captures[-1] < c.min_gap_s:
                ignored = 'observations_not_independent'
        # Ignored evidence can advance time, but never postpone expiry/probes.
        if ignored != 'context_mismatch':
            self._now = max(self._now, o.observed_at)
        if ignored is not None:
            result = self._result(o.pose, self._now)
            result.update(accepted=False, ignored_reason=ignored)
            return result
        if record is None:
            if len(self._records) >= c.max_poses:
                oldest = min(self._records, key=lambda p: (self._records[p]['last_observed_at'], p))
                del self._records[oldest]
            record = {'last_captured_at': o.captured_at, 'last_observed_at': o.observed_at,
                      'high_captures': [], 'excluded_until': None,
                      'next_probe_at': o.observed_at + c.probe_interval_s, 'reason': 'uninspected_view'}
            self._records[o.pose] = record
        record['last_captured_at'], record['last_observed_at'] = o.captured_at, o.observed_at
        if o.pet_detected or o.obstruction_probability <= c.low_threshold:
            record.update(high_captures=[], excluded_until=None,
                          next_probe_at=o.observed_at + c.probe_interval_s,
                          reason='pet_seen_clears_obstruction' if o.pet_detected else 'clear_visual_evidence')
        elif o.obstruction_probability < c.high_threshold:
            record.update(high_captures=[], excluded_until=None,
                          next_probe_at=o.observed_at + c.probe_interval_s,
                          reason='ambiguous_visual_evidence_fail_open')
        else:
            captures = [t for t in record['high_captures'] if o.captured_at - t <= c.evidence_window_s]
            captures.append(o.captured_at)
            if len(captures) > 256:
                captures = [captures[0]] + captures[-255:]
            record['high_captures'] = captures
            enough = len(captures) >= c.min_samples and captures[-1] - captures[0] >= c.min_span_s
            if enough:
                record.update(excluded_until=o.observed_at + c.exclusion_ttl_s,
                              next_probe_at=o.observed_at + c.probe_interval_s,
                              reason='repeated_independent_visual_obstruction')
            else:
                record.update(excluded_until=None, next_probe_at=o.observed_at + c.min_gap_s,
                              reason='insufficient_independent_visual_evidence')
        result = self._result(o.pose, self._now)
        result.update(accepted=True, ignored_reason=None)
        return result

    def due(self, poses, now, context, limit):
        """Propose at most limit probes; repeat calls never consume the budget.

        Mandatory reinspection precedes suspected and unknown exploration.
        Unattempted/oldest-attempt views precede recently attempted views, then
        deadline and pose break ties. Invalid probes rotate this proposal order
        while remaining due and fail open. Read-only calls never count as an
        attempt, and only accepted visual evidence updates deadlines.
        """
        if type(limit) is not int or limit < 0:
            raise ValueError('limit must be a nonnegative integer')
        poses = sorted(set(_pose(p) for p in poses))
        context = _context(context)
        now = self._time(now)
        self._activate(context, now)
        candidates = [self._result(p, now) for p in poses]
        ranks = {'reinspection_due': 0, 'suspected': 1, 'unknown': 2}
        candidates = [d for d in candidates if d['probe_due']]
        return sorted(candidates, key=lambda d: (ranks[d['state']],
                                                -1 if d['last_probe_attempt_at'] is None else d['last_probe_attempt_at'],
                                                d['probe_deadline'], d['pose']))[:limit]

    def to_dict(self):
        """Return a detached JSON-safe snapshot, including the active setup."""
        return {'version': VERSION, 'config': asdict(self.config), 'context': self._context,
                'context_since': self._context_since, 'now': self._now,
                'probe_attempts': [{'pose': list(p), **r} for p, r in sorted(self._probe_attempts.items())],
                'records': [{'pose': list(p), **{k: list(v) if isinstance(v, list) else v for k, v in r.items()}}
                            for p, r in sorted(self._records.items())]}

    @classmethod
    def from_dict(cls, snapshot):
        """Restore a validated snapshot; live queries still verify context/TTL."""
        try:
            if snapshot['version'] != VERSION:
                raise ValueError('unsupported obstruction memory version')
            memory = cls(ObstructionConfig(**snapshot['config']))
            memory._now = _number(snapshot['now'], 'now')
            memory._context_since = _number(snapshot['context_since'], 'context_since', maximum=memory._now)
            memory._context = None if snapshot['context'] is None else _context(snapshot['context'])
            records = snapshot['records']
            if not isinstance(records, list) or len(records) > memory.config.max_poses:
                raise ValueError('invalid obstruction records')
            if records and memory._context is None:
                raise ValueError('records require an active context')
            for raw in records:
                pose = _pose(raw['pose'])
                if pose in memory._records:
                    raise ValueError('duplicate saved pose')
                captured = _number(raw['last_captured_at'], 'last_captured_at', maximum=memory._now)
                observed = _number(raw['last_observed_at'], 'last_observed_at', minimum=captured, maximum=memory._now)
                if captured < memory._context_since or observed - captured > memory.config.max_frame_age_s:
                    raise ValueError('saved evidence predates setup or was stale')
                times = [_number(t, 'high capture', minimum=memory._context_since, maximum=captured) for t in raw['high_captures']]
                if len(times) > 256 or any(b-a < memory.config.min_gap_s for a, b in zip(times, times[1:])):
                    raise ValueError('saved high observations are not independent')
                if times and captured - times[0] > memory.config.evidence_window_s:
                    raise ValueError('saved high observations exceed the evidence window')
                expiry = raw['excluded_until']
                expiry = None if expiry is None else _number(expiry, 'excluded_until', minimum=observed,
                                                              maximum=observed + memory.config.exclusion_ttl_s)
                deadline = _number(raw['next_probe_at'], 'next_probe_at', minimum=observed,
                                   maximum=observed + max(memory.config.probe_interval_s, memory.config.min_gap_s))
                if expiry is not None and (len(times) < memory.config.min_samples or
                                          times[-1] - times[0] < memory.config.min_span_s or times[-1] != captured):
                    raise ValueError('saved exclusion lacks independent evidence')
                if not isinstance(raw['reason'], str):
                    raise ValueError('invalid saved reason')
                memory._records[pose] = {'last_captured_at': captured, 'last_observed_at': observed,
                                        'high_captures': times, 'excluded_until': expiry,
                                        'next_probe_at': deadline, 'reason': raw['reason']}
            # Snapshots written before attempt scheduling remain readable.
            attempts = snapshot.get('probe_attempts', [{'pose': list(p), 'captured_at': r['last_captured_at'],
                                                       'observed_at': r['last_observed_at']}
                                                      for p, r in memory._records.items()])
            if not isinstance(attempts, list) or len(attempts) > memory.config.max_poses:
                raise ValueError('invalid saved probe attempts')
            if attempts and memory._context is None:
                raise ValueError('probe attempts require an active context')
            for raw in attempts:
                pose = _pose(raw['pose'])
                if pose in memory._probe_attempts:
                    raise ValueError('duplicate saved probe attempt')
                captured = _number(raw['captured_at'], 'attempt captured_at', minimum=memory._context_since,
                                   maximum=memory._now)
                observed = _number(raw['observed_at'], 'attempt observed_at', minimum=captured, maximum=memory._now)
                if observed - captured > memory.config.max_frame_age_s:
                    raise ValueError('saved probe attempt was stale')
                memory._probe_attempts[pose] = {'captured_at': captured, 'observed_at': observed}
            return memory
        except (KeyError, TypeError, OverflowError):
            raise ValueError('invalid obstruction memory snapshot') from None

    def save(self, path):
        """Explicit persistence only; does not create parent directories."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + '\n')

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
