"""Gymnasium environment for learning Follow on the calibrated simulator (sim/room.py, sim/params.json).

One step = one watcher cycle, run exactly as sim/gate1.py runs the real controller (same detector schedule, same stale
results between detector runs, 0.7 s frame lag, the measured motor). The policy sees only what the real watcher has.
Ground truth (is she really in the frame, is she home) is used for the reward and never for the observation.

Actions: MultiDiscrete([9, 2]):
  motor  0 wait · 1 centre on the latest sighting · 2 lead-aim (sighting + velocity x 1 s) · 3/4 turn 0.75 frame
         left/right · 5/6 tilt 0.6 frame up/down · 7 search (the real routine from server/tracker.py) · 8 stop
  detect 0 normal schedule (server/watch.py) · 1 fast: a cycle and a detector run every 0.5 s (costs CPU)
Shield: while she was seen lying still in the last 8 s, every motor action except wait/stop becomes wait. The
owner's rule ("no automatic motor moves near a resting dog") is enforced in code, so no policy can trade it away.
Reward per step: seconds she is truly in view while home, minus MOVE_COST per move and MOVE_FRAME_COST per frame width
moved, minus FAST_COST per second of fast mode. Episode: one simulated hour.
"""
import os
import sys

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate1  # noqa: E402
from room import Room, load_params  # noqa: E402

MOVE_COST, MOVE_FRAME_COST, FAST_COST = 1.0, 0.5, 0.15
EPISODE_S = 3600.0
STILL_SPAN, STILL_SPREAD, REST_HOLD = 3.0, 0.06, 8.0
MOVE_BUDGET, BUDGET_WINDOW = 6, 600.0   # hard limit: at most 6 moves per 10 minutes (today's tracker makes ~31-44/h)


class FollowEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None, free=None, randomize=False, episode_s=EPISODE_S):
        super().__init__()
        self.P = load_params()
        self.free = free if free is not None else self.P.get("free")
        self.randomize = randomize
        self.episode_s = episode_s
        self.action_space = gym.spaces.MultiDiscrete([9, 2])
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(18,), dtype=np.float32)
        self.T = gate1.load_tracker()
        self._seed = seed

    # domain randomisation around the calibrated parameters (x0.6..1.6 on rates and speeds)
    def _draw_free(self, rng):
        f = dict(self.free or {})
        if self.randomize:
            for k in ("walk_speed_med", "rest_med_s", "away_med_s", "people_motion_per_h"):
                if k in f:
                    f[k] = f[k] * float(rng.uniform(0.6, 1.6))
            if "away_prob" in f:
                f["away_prob"] = float(np.clip(f["away_prob"] * rng.uniform(0.6, 1.4), 0.02, 0.8))
            f["lag_s"] = float(rng.uniform(0.5, 1.1))
        return f

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = int(self.np_random.integers(0, 2**31 - 1)) if seed is None else seed
        rng = np.random.default_rng(s)
        if options and "room" in options:          # an explicit (seed, free) room: the shared held-out bank
            s, free = options["room"]
            self.room = Room(seed=s, free=free)
        else:
            self.room = Room(seed=s, free=self._draw_free(rng))
        self.tr = self.T.Tracker(self.room, self.room.find_pet, clock=lambda: self.room.t, log=lambda m: None,
                                 sleep=self.room.sleep, find_now=self.room.find_now)
        self.t0 = self.room.t
        self.last_detect, self.last_motion_t = -1e9, -1e9
        self.result = (None, False, [])
        self.sightings = []                  # (t, cx, cy) of fresh sightings, frame coords centred on 0
        self.last_move_t, self.fast = -1e9, 0
        self.stats = {"moves": 0, "shielded": 0, "fast_s": 0.0, "in_view_s": 0.0, "home_s": 0.0}
        self._tick()
        return self._obs(), {}

    def _lay_still(self):
        h = [s for s in self.sightings if self.room.t - s[0] <= 4.0]
        if len(h) < 2 or h[-1][0] - h[0][0] < STILL_SPAN:
            return False
        xs, ys = [s[1] for s in h], [s[2] for s in h]
        return max(xs) - min(xs) <= STILL_SPREAD and max(ys) - min(ys) <= STILL_SPREAD

    def _tick(self):
        """One watcher cycle of sensing (server/watch.py): maybe run the detector, else repeat the last result."""
        r = self.room
        self.motion = r.motion()
        if self.motion:
            self.last_motion_t = r.t
        seen_recent = self.sightings and r.t - self.sightings[-1][0] < 20
        lively = r.t - self.last_motion_t < 60 or self.tr.searching or seen_recent
        self.fresh = bool(self.fast or self.motion or r.t - self.last_detect >= (8.0 if lively else 30.0))
        if self.fresh:
            self.result, self.last_detect = r.frame(), r.t
            box = self.result[0]
            if box is not None:
                self.sightings.append((r.t, (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5))
                self.sightings = self.sightings[-20:]

    def _velocity(self):
        h = [s for s in self.sightings if self.room.t - s[0] <= 4.0]
        if len(h) < 2 or h[-1][0] - h[0][0] < 0.5:
            return 0.0, 0.0
        dt = h[-1][0] - h[0][0]
        return (h[-1][1] - h[0][1]) / dt, (h[-1][2] - h[0][2]) / dt

    def _obs(self):
        r = self.room
        box = self.result[0]
        last = self.sightings[-1] if self.sightings else None
        age = (r.t - last[0]) if last else 999.0
        vx, vy = self._velocity()
        m = self.motion
        o = [float(self.fresh), float(box is not None),
             (last[1] if last else 0.0), (last[2] if last else 0.0),
             min(age, 120.0) / 30.0, float(np.clip(vx, -2, 2)), float(np.clip(vy, -2, 2)),
             r.X / 1850.0 - 1.0, r.Y / 500.0 - 1.0, float(r.busy()),
             float(m is not None), (m[1] if m else 0.0), (m[2] if m else 0.0),
             min(len(self.result[2]), 3) / 3.0, float(self.fast), min(r.t - self.last_move_t, 60.0) / 30.0,
             float(self._lay_still()),
             1.0 - min(1.0, sum(1 for m in r.moves if r.t - m[0] < BUDGET_WINDOW) / MOVE_BUDGET)]
        return np.asarray(o, dtype=np.float32)

    def _act(self, a):
        r = self.room
        if a in (0, 8):
            if a == 8:
                r.stop()
            return
        if self._lay_still() and self.sightings and r.t - self.sightings[-1][0] < REST_HOLD:
            self.stats["shielded"] += 1
            return
        if sum(1 for m in r.moves if r.t - m[0] < BUDGET_WINDOW) >= MOVE_BUDGET:
            self.stats["budget_blocked"] = self.stats.get("budget_blocked", 0) + 1     # a hard constraint, not a cost
            return
        last = self.sightings[-1] if self.sightings else None
        n0 = len(r.moves)
        if a in (1, 2):
            if not last or r.t - last[0] > 3.0:
                return
            vx, vy = self._velocity() if a == 2 else (0.0, 0.0)
            lag = r.t - last[0] + 1.0
            r.aim(last[1] + vx * lag, last[2] + vy * lag)
        elif a in (3, 4):
            r.aim(-0.75 if a == 3 else 0.75, 0.0)
        elif a in (5, 6):
            r.aim(0.0, -0.6 if a == 5 else 0.6)
        elif a == 7:
            self.tr.search("rl")
        if len(r.moves) > n0:
            self.last_move_t = r.t

    def step(self, action):
        motor, detect = int(action[0]), int(action[1])
        r = self.room
        t_start, moves0 = r.t, len(r.moves)
        truth0 = len(r.truth)
        self.fast = detect
        self._act(motor)
        period = 0.5 if self.fast else (1.0 if self.motion else 3.0)
        r.advance(max(0.05, period - (r.t - t_start)))
        self._tick()
        # reward from ground truth over the time this step covered
        seg = r.truth[truth0:]
        in_view = sum(dt for _, dt, v, h in seg if v and h)
        home = sum(dt for _, dt, _, h in seg if h)
        new_moves = r.moves[moves0:]
        moved_frames = sum(m[2] for m in new_moves) / r.FW
        dur = r.t - t_start
        reward = in_view - MOVE_COST * len(new_moves) - MOVE_FRAME_COST * moved_frames - FAST_COST * dur * self.fast
        self.stats["moves"] += len(new_moves)
        self.stats["fast_s"] += dur * self.fast
        self.stats["in_view_s"] += in_view
        self.stats["home_s"] += home
        truncated = r.t - self.t0 >= self.episode_s
        if len(r.truth) > 200000:
            r.truth = r.truth[-1000:]
        info = dict(self.stats) if truncated else {}
        return self._obs(), float(reward) / 60.0, False, truncated, info


def episode_score(stats, hours):
    """The comparison everyone is judged by: true in-view share while home, moves/hour, fast-mode minutes/hour."""
    return {"in_view": round(stats["in_view_s"] / max(stats["home_s"], 1e-9), 3),
            "moves_per_h": round(stats["moves"] / hours, 1),
            "fast_min_per_h": round(stats["fast_s"] / 60 / hours, 1),
            "shielded": stats["shielded"]}
