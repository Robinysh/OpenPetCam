"""Controlled, offline search scenes for the real Tracker camera interface.

Target motion is a function of time, and detector draws are keyed by seed and
frame number. Looking more often cannot change either. This deliberately small
angular model does not claim complete room geometry or detector realism.
"""
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Optional, Tuple

try:
    from .room import Room
except ImportError:  # direct use from a script with sim on sys.path
    from room import Room


Pose = Tuple[float, float]


@dataclass(frozen=True)
class SearchScenario:
    seed: int = 0
    target_start: Pose = (1850.0, 500.0)
    target_end: Optional[Pose] = None
    move_start_s: float = 0.0
    move_duration_s: float = 0.0
    absent: bool = False
    occluded: bool = False
    detection_p: float = 1.0
    false_positive_p: float = 0.002
    lag_s: float = 0.9
    start_pose: Pose = (1019.0, 534.0)
    budget_s: Optional[float] = None
    box_jitter: float = 0.0

    def __post_init__(self):
        for name in ("target_start", "target_end", "start_pose"):
            value = getattr(self, name)
            if value is None and name == "target_end":
                value = self.target_start
            if len(value) != 2 or not all(math.isfinite(float(x)) for x in value):
                raise ValueError(f"{name} must contain two finite coordinates")
            object.__setattr__(self, name, tuple(float(x) for x in value))
        for name in ("move_start_s", "move_duration_s", "lag_s", "box_jitter"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("detection_p", "false_positive_p"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a probability")
        if self.budget_s is not None and (not math.isfinite(self.budget_s) or self.budget_s < 0):
            raise ValueError("budget_s must be finite and nonnegative")


@dataclass(frozen=True)
class FrameEvidence:
    frame_index: int
    arrived_at: float
    captured_at: float
    camera_pose: Pose                  # optical pose at exposure, not the current counter
    target_pose: Pose
    present: bool
    occluded: bool
    in_view: bool
    true_detection: bool
    false_detection: bool
    box: Optional[Tuple[float, float, float, float]]


class DetectionBox(list):
    """List-compatible tracker box carrying its immutable exposure-time truth."""

    def __init__(self, evidence):
        super().__init__(evidence.box)
        self.evidence = evidence
        self.captured = evidence.captured_at


class SearchDeadline(RuntimeError):
    def __init__(self, deadline_s):
        self.deadline_s = deadline_s
        super().__init__(f"offline search budget exhausted at {deadline_s:g} s")


class SearchScene(Room):
    """Room motor physics with a predeclared target trajectory and frame sensor.

``actual_travel_steps`` integrates absolute pan plus tilt counter movement,
including interrupted moves. ``pose_trace`` retains every physics sample as
``(time, pan_counter, tilt, optical_pan)``. Commanded distances in Room.moves
remain available but are not used as actual travel.
    """

    def __init__(self, scenario, params=None, free=None, presets=None, budget_s=None):
        if isinstance(scenario, Mapping):
            scenario = SearchScenario(**scenario)
        if not isinstance(scenario, SearchScenario):
            raise TypeError("scenario must be SearchScenario or a mapping of its fields")
        self.scenario = scenario
        controlled = dict(free or {}, lag_s=scenario.lag_s,
                          owner_moves_per_h=0.0, people_motion_per_h=0.0,
                          p_false_accept=scenario.false_positive_p)
        super().__init__(seed=scenario.seed, params=params, free=controlled, presets=presets)
        x, y = scenario.start_pose
        if not 0 <= x <= self.XM or not 40 <= y <= self.YM - 40:
            raise ValueError("start_pose is outside camera motor limits")
        self.C = self.X = x
        self.Y = y
        self.cam_hist = [(0.0, x, y)]
        self.next_owner = self.next_people = math.inf
        self._dog_step(0.0)
        self.dog_hist = [(0.0, self.px, self.py, self.dog_state == "move", self.dog_state, self.hidden)]
        self.pose_trace = [(0.0, self.C, self.Y, self.X)]
        self.world_trace = [(0.0, self.px, self.py, not scenario.absent, scenario.occluded)]
        self._pose_times = [0.0]
        self.actual_travel_steps = 0.0
        self.actual_optical_travel_steps = 0.0
        self.pan_travel_steps = self.tilt_travel_steps = 0.0
        self._moving_commands = set()
        self._active_command = None
        self.detection_evidence = []
        self.accepted_detections = []
        self.last_frame_evidence = self.last_detection_evidence = None
        self.last_detection_true = None
        self.budget_s = scenario.budget_s if budget_s is None else budget_s
        if self.budget_s is not None and (not math.isfinite(self.budget_s) or self.budget_s < 0):
            raise ValueError("budget_s must be finite and nonnegative")

    def target_at(self, t):
        s = self.scenario
        if t < s.move_start_s:
            fraction = 0.0
        elif s.move_duration_s == 0:
            fraction = 1.0
        else:
            fraction = min(1.0, max(0.0, (t - s.move_start_s) / s.move_duration_s))
        return tuple(a + fraction * (b - a) for a, b in zip(s.target_start, s.target_end))

    def _dog_step(self, dt):
        # Room calls this before incrementing its clock. No random draws or
        # previous target state participate in this update.
        t = self.t + dt
        s = self.scenario
        self.px, self.py = self.target_at(t)
        moving = s.move_duration_s > 0 and s.move_start_s <= t < s.move_start_s + s.move_duration_s
        self.dog_state = "away" if s.absent else ("move" if moving else "rest")
        self.hidden = s.occluded
        self.vx = self.vy = 0.0

    def _start(self, x, y, kind):
        super()._start(x, y, kind)
        self._active_command = len(self.moves) - 1

    @property
    def actual_move_count(self):
        return len(self._moving_commands)

    def advance(self, dt):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("advance duration must be finite and nonnegative")
        requested_end = self.t + dt
        if self.budget_s is not None and self.t >= self.budget_s - 1e-10:
            self.stop()
            raise SearchDeadline(self.budget_s)
        end = min(requested_end, self.budget_s) if self.budget_s is not None else requested_end
        normal_dt = self.dt
        try:
            while end - self.t > 1e-10:
                # Room rounds dt to its integration grid. A final short step
                # makes the deadline and observation boundaries exact instead.
                self.dt = min(normal_dt, end - self.t)
                c0, y0, x0 = self.C, self.Y, self.X
                super().advance(self.dt)
                dc, dy, dx = abs(self.C - c0), abs(self.Y - y0), abs(self.X - x0)
                self.pan_travel_steps += dc
                self.tilt_travel_steps += dy
                self.actual_travel_steps += dc + dy
                self.actual_optical_travel_steps += dx + dy
                if dc + dy > 1e-10 and self._active_command is not None:
                    self._moving_commands.add(self._active_command)
                self.pose_trace.append((self.t, self.C, self.Y, self.X))
                self._pose_times.append(self.t)
                self.world_trace.append((self.t, self.px, self.py, not self.scenario.absent, self.scenario.occluded))
        finally:
            self.dt = normal_dt
        if self.budget_s is not None and requested_end >= self.budget_s - 1e-10:
            self.stop()
            raise SearchDeadline(self.budget_s)

    def _camera_at(self, t):
        if t <= 0:
            return self.pose_trace[0][3], self.pose_trace[0][2]
        j = min(len(self.pose_trace) - 1, bisect_right(self._pose_times, t) - 1)
        a = self.pose_trace[j]
        if j == len(self.pose_trace) - 1:
            return a[3], a[2]
        b = self.pose_trace[j + 1]
        f = min(1.0, max(0.0, (t - a[0]) / (b[0] - a[0])))
        return a[3] + f * (b[3] - a[3]), a[2] + f * (b[2] - a[2])

    def _draw(self, frame_index, channel):
        key = f"search-scene-v1:{self.scenario.seed}:{frame_index}:{channel}".encode("ascii")
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / (1 << 64)

    def frame(self):
        # A frame's arrival is on the same 5 fps grid as Room.find_pet/find_now.
        # Constant lag gives a one-to-one frame index for exposure and arrival.
        index = int(math.floor(self.t / self.FRAME_DT + 1e-8))
        arrival = index * self.FRAME_DT
        captured = arrival - self.scenario.lag_s
        X, Y = self._camera_at(captured)
        px, py = self.target_at(captured)
        s = self.scenario
        f, g = self.offset(X, Y, px, py)
        visible = not s.absent and not s.occluded and abs(f) < 0.45 and abs(g) < 0.45
        true = visible and self._draw(index, "detect") < s.detection_p
        false = not true and self._draw(index, "false") < s.false_positive_p
        box = None
        if true:
            jitter = [s.box_jitter * (2 * self._draw(index, f"jitter-{k}") - 1) for k in range(4)]
            box = tuple(v + j for v, j in zip((0.5 + f - 0.08, 0.5 + g - 0.1,
                                               0.5 + f + 0.08, 0.5 + g + 0.1), jitter))
        elif false:
            cx = 0.2 + 0.6 * self._draw(index, "false-x")
            cy = 0.3 + 0.5 * self._draw(index, "false-y")
            box = (cx - 0.06, cy - 0.08, cx + 0.06, cy + 0.08)
        evidence = FrameEvidence(index, arrival, captured, (X, Y), (px, py),
                                 not s.absent, s.occluded, visible, true, false, box)
        self.last_frame_evidence = evidence
        self.detection_evidence.append(evidence)
        if box is not None:
            self.last_detection_evidence = evidence
            self.last_detection_true = true
            self.accepted_detections.append(evidence)
        moving = s.move_duration_s > 0 and s.move_start_s <= captured < s.move_start_s + s.move_duration_s
        return DetectionBox(evidence) if box is not None else None, bool(true and moving), []

    def find_now(self, since):
        newest = math.floor(self.t / self.FRAME_DT + 1e-9) * self.FRAME_DT
        if newest <= since or newest <= getattr(self, "last_judged", -1.0):
            return None
        self.last_judged = newest
        return self.frame()[0]  # preserve DetectionBox.evidence and captured

    def motion(self):
        captured = math.floor(self.t / self.FRAME_DT + 1e-8) * self.FRAME_DT - self.scenario.lag_s
        s = self.scenario
        if s.absent or s.occluded or not s.move_start_s <= captured < s.move_start_s + s.move_duration_s:
            return None
        X, Y = self._camera_at(captured)
        f, g = self.offset(X, Y, *self.target_at(captured))
        return (captured, max(-0.49, min(0.49, f)), max(-0.49, min(0.49, g))) if abs(f) < 0.6 and abs(g) < 0.6 else None
