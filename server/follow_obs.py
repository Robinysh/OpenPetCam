"""What a learned Follow policy sees, computed the same way in training (sim/env.py) and on the camera (server/watch.py).

One class, shared, so the server can never feed the policy an observation built differently from the one it was trained
on. 18 numbers per watcher cycle, in this order:
  fresh (this cycle ran the detector), box now, last sighting x/y offset (frame units, 0 = centre), age of that sighting
  /30 s, her velocity x/y (frame widths/s over 4 s), pan/1850-1, tilt/500-1, motor busy, motion now, motion x/y, weak
  boxes /3, fast mode, seconds since our last move /30, lying still, move budget left (0..1).
The two hard rules live here too: no move while she was seen lying still in the last REST_HOLD s, and at most
MOVE_BUDGET moves per BUDGET_WINDOW s.
"""
import numpy as np

STILL_SPAN, STILL_SPREAD, REST_HOLD = 3.0, 0.06, 8.0
MOVE_BUDGET, BUDGET_WINDOW = 6, 600.0
SEARCH_UNSEEN = 10.0     # a search needs her unseen this long (live smoke test 2026-11-04: PPO had learned to "search"
                         # every cycle while she was in view, as a free 4-frame look; 183 of 186 cycles)
N_OBS = 18


class FollowObs:
    def __init__(self):
        self.sightings = []          # (t, cx, cy) of fresh sightings, frame coordinates centred on 0
        self.move_times = []         # when our own moves started (for the budget)
        self.last_move_t = -1e9
        self.fresh, self.box, self.motion, self.weak_n, self.fast = False, None, None, 0, 0

    def update(self, t, fresh, box, motion, weak_n, fast=0):
        """One watcher cycle: fresh = the detector ran this cycle; box = its verified box (or None)."""
        self.fresh, self.box, self.motion, self.weak_n, self.fast = bool(fresh), box, motion, int(weak_n), int(fast)
        if fresh and box is not None:
            self.sightings.append((t, (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5))
            self.sightings = self.sightings[-20:]

    def note_move(self, t):
        """A move that started at t (budget). The caller sets last_move_t to when its whole action finished."""
        self.move_times.append(t)
        self.move_times = [m for m in self.move_times if t - m < 3600]
        self.last_move_t = t

    def lay_still(self, t):
        h = [s for s in self.sightings if t - s[0] <= 4.0]
        if len(h) < 2 or h[-1][0] - h[0][0] < STILL_SPAN:
            return False
        xs, ys = [s[1] for s in h], [s[2] for s in h]
        return max(xs) - min(xs) <= STILL_SPREAD and max(ys) - min(ys) <= STILL_SPREAD

    def velocity(self, t):
        h = [s for s in self.sightings if t - s[0] <= 4.0]
        if len(h) < 2 or h[-1][0] - h[0][0] < 0.5:
            return 0.0, 0.0
        dt = h[-1][0] - h[0][0]
        return (h[-1][1] - h[0][1]) / dt, (h[-1][2] - h[0][2]) / dt

    def budget_used(self, t):
        return sum(1 for m in self.move_times if t - m < BUDGET_WINDOW)

    def blocked(self, t, motor=None):
        """Why a motor action may not run now: 'seen' (a search while she was just seen), 'rest' (she lies still in
        view), 'budget', or None. motor 7 = search."""
        if motor == 7 and self.sightings and t - self.sightings[-1][0] < SEARCH_UNSEEN:
            return "seen"
        if self.lay_still(t) and self.sightings and t - self.sightings[-1][0] < REST_HOLD:
            return "rest"
        if self.budget_used(t) >= MOVE_BUDGET:
            return "budget"
        return None

    def vector(self, t, pan, tilt, busy):
        last = self.sightings[-1] if self.sightings else None
        age = (t - last[0]) if last else 999.0
        vx, vy = self.velocity(t)
        m = self.motion
        o = [float(self.fresh), float(self.box is not None),
             (last[1] if last else 0.0), (last[2] if last else 0.0),
             min(age, 120.0) / 30.0, float(np.clip(vx, -2, 2)), float(np.clip(vy, -2, 2)),
             pan / 1850.0 - 1.0, tilt / 500.0 - 1.0, float(busy),
             float(m is not None), (m[1] if m else 0.0), (m[2] if m else 0.0),
             min(self.weak_n, 3) / 3.0, float(self.fast), min(t - self.last_move_t, 60.0) / 30.0,
             float(self.lay_still(t)), 1.0 - min(1.0, self.budget_used(t) / MOVE_BUDGET)]
        return np.asarray(o, dtype=np.float32)
