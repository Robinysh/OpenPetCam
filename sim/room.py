"""The Follow simulator: one room, one pan/tilt camera, one dog, fitted to the real install (sim/params.json).

A single physics core (Room) drives both the real controller (server/tracker.py, through the same camera API as the
watcher gives it) and the reinforcement-learning environment (sim/env.py). Everything random comes from one seeded
numpy Generator, so a seed reproduces a run exactly.

What is measured (sim/params.json, sim/fit_from_logs.py, 2026-11-03/04):
  camera   218 steps/s at the quiet speed, ~0.1 s to start, a new command waits for the move in progress
           (the app waits for idle, then moves); one frame = 733 x 424 steps; pan 0..3700, tilt 0..1000
  sensing  frames arrive 0.7 s late; the watcher cycles every ~3 s, or ~1 s while motion wakes it;
           P(seen next cycle | seen now) 0.95 moving, 0.988 still
  dog      where she is seen (pan hotspots), rest and move dwell times
What is NOT directly observable and is calibrated against the logged statistics instead (gate 1, sim/gate1.py):
  her true walking speed, how long she is away, how often people set off motion. Those are the `free` parameters.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FREE_DEFAULTS = {
    "walk_speed_med": 120.0,      # steps/s, log-normal median of a walk (logs only see the slow, in-frame part)
    "walk_speed_sigma": 0.7,
    "rest_med_s": 240.0,          # how long she lies in one place
    "rest_sigma": 1.2,
    "away_prob": 0.25,            # a walk that ends out of the camera's reach (bedroom, a walk outside)
    "away_med_s": 900.0,
    "away_sigma": 1.0,
    "people_motion_per_h": 6.0,   # motion events from people that look like her leaving (no dog in them)
    "p_false_accept": 0.002,      # per judged frame: the verifier passes something that is not her
    "lag_s": 0.7,
    "owner_moves_per_h": 0.0,     # people turning the camera by hand (logged as 'paused'); fitted directly from logs
    "p_hidden": 0.3,              # lying down inside an occluder (sim/params.json scene, mapped by Claude): hidden for
                                  # that rest with this probability (calibrated)
    "rest_place_share": 0.2,      # share of lie-downs at Claude's rest places rather than the logged sighting map
    # Hard poses (2026-11-04 search audit): lying flat or curled, the detector's raw score sits at its 0.10 floor and it
    # proposes her on only some frames, while the verifier is sure whenever it does. detect_persist (0.99 per frame)
    # is measured while the watcher is already tracking her, i.e. on the poses it sees well; the losses that start a
    # search are the others. Each lie-down is hard with q_hard, and then each frame shows her with p_hard (calibrated
    # to the real search outcomes). Defaults keep the old behaviour.
    "q_hard": 0.0,
    "p_hard": 0.3,
}


def load_params(path=os.path.join(HERE, "params.json")):
    return json.load(open(path))


class _Seen(list):
    """A box with the time its frame was taken (the watcher's find_pet_now gives the tracker the same)."""


class Room:
    def __init__(self, seed=0, params=None, free=None, presets=None):
        self.P = params or load_params()
        self.F = dict(FREE_DEFAULTS, **(free or {}))
        g = self.P["geometry"]
        self.XM, self.YM, self.FW, self.FH = g["pan_max"], g["tilt_max"], g["pan_frame"], g["tilt_frame"]
        if (free or {}).get("pan_frame"):                # a measured range is randomised per room (sim/train_policy.py)
            self.FW = float(free["pan_frame"])
        self.speed = float(self.P["motor"]["speed_steps_s"]["250"])
        self.rng = np.random.default_rng(seed)
        # Sensing draws from its own stream (review 2026-11-04): with one shared stream, a policy that looks more often
        # changed where the dog went next under the same seed, so paired policy comparisons were not paired.
        self.rng_sense = np.random.default_rng([seed, 7919])
        self.dt = 0.05
        self.t = 0.0
        # camera
        self.X, self.Y = 1019.0, 534.0                  # where the picture points (optical), 'Corridor' rest view
        self.C = self.X                                 # the pan step counter (motors -j); X follows it with gear play
        self.target = None                              # (x, y, start_time)
        self.backlash = float((free or {}).get("pan_backlash") or g.get("pan_backlash", 0.0))   # gear play
        self.cam_hist = [(0.0, self.X, self.Y)]
        self.moved = -1e9
        self.moved_by_owner = -1e9
        self.moves = []                                 # (t, kind, steps moved)
        self._presets = presets if presets is not None else [
            {"name": "Next to TV", "x": 1718, "y": 426}, {"name": "Door", "x": 355, "y": 450},
            {"name": "Corridor", "x": 1019, "y": 534}, {"name": "Couch", "x": 2252, "y": 704},
            {"name": "Carpet", "x": 1952, "y": 577}]
        # dog: state is 'rest' | 'move' | 'away'
        hs = self.P["pet"]["pan_hotspots"]
        self.hot_x = np.array([float(k) + self.FW / 4 for k in hs])
        self.hot_p = np.array(list(hs.values()), dtype=float)
        self.hot_p /= self.hot_p.sum()
        self.dog_state, self.px, self.py = "rest", *self._spot()
        self.state_until = self.t + self._lognorm(self.F["rest_med_s"], self.F["rest_sigma"])
        self.goal = None
        self.vx = self.vy = 0.0
        self.hidden = False
        self.leaving = False
        self.dog_hist = [(0.0, self.px, self.py, False, self.dog_state, False)]
        self.next_people = self.t + self.rng.exponential(3600 / max(self.F["people_motion_per_h"], 1e-6))
        om = float(self.F.get("owner_moves_per_h") or 0.0)
        self.next_owner = self.t + (self.rng.exponential(3600 / om) if om > 0 else 1e18)
        self.people_motion = None                       # (t, cx, cy) of the latest people motion
        self.truth = []                                 # (t, dt, in_view, home) for ground-truth metrics

    # ---------------------------------------------------------------- dog
    def _lognorm(self, med, sigma):
        return float(med * math.exp(self.rng.normal(0, sigma)))

    def _scene(self, kind):
        return (self.P.get("scene") or {}).get(kind) or []

    def _inside(self, kind, x, y):
        return [r for r in self._scene(kind) if r["pan"][0] <= x <= r["pan"][1] and r["tilt"][0] <= y <= r["tilt"][1]]

    def _spot(self):
        """A place she lies down: Claude's rest places (a share), else the logged 2D map of where she is seen."""
        rp = self._scene("rest_places")
        if rp and self.rng.random() < float(self.F.get("rest_place_share", 0.0)):
            r = rp[int(self.rng.integers(len(rp)))]
            return (float(self.rng.uniform(*r["pan"])),
                    min(self.YM - 40.0, max(40.0, float(self.rng.uniform(*r["tilt"])))))
        rm = self.P["pet"].get("room_map")
        if rm:
            w = np.array([r[2] for r in rm], dtype=float)
            r = rm[int(self.rng.choice(len(rm), p=w / w.sum()))]
            x = float(r[0] + self.rng.uniform(-self.FW / 4, self.FW / 4))
            y = float(r[1] + self.rng.uniform(-self.FH / 4, self.FH / 4))
            return min(self.XM, max(0.0, x)), min(self.YM - 40.0, max(40.0, y))
        x = float(self.rng.choice(self.hot_x, p=self.hot_p) + self.rng.normal(0, self.FW / 6))
        return min(self.XM, max(0.0, x)), float(self.rng.uniform(420, 720))

    def _dog_step(self, dt):
        if self.t >= self.state_until:
            if self.dog_state == "rest":
                self.dog_state = "move"
                self.hidden = False
                away = self.rng.random() < self.F["away_prob"]
                ex = self._scene("exits")
                if away and ex:                              # she leaves through a real doorway (Claude's map)
                    e = ex[int(self.rng.integers(len(ex)))]
                    self.goal = (float(sum(e["pan"]) / 2), float(sum(e["tilt"]) / 2))
                    self.leaving = True
                else:
                    self.goal = ((-self.FW if self.rng.random() < 0.5 else self.XM + self.FW), self.py) if away else self._spot()
                sp = self._lognorm(self.F["walk_speed_med"], self.F["walk_speed_sigma"])
                d = max(1.0, math.hypot(self.goal[0] - self.px, self.goal[1] - self.py))
                self.vx, self.vy = sp * (self.goal[0] - self.px) / d, sp * (self.goal[1] - self.py) / d
                self.state_until = self.t + d / sp
            elif self.dog_state == "move":
                if getattr(self, "leaving", False) or self.px < 0 or self.px > self.XM:
                    self.leaving = False
                    self.dog_state = "away"
                    self.state_until = self.t + self._lognorm(self.F["away_med_s"], self.F["away_sigma"])
                else:
                    self.dog_state = "rest"
                    self.state_until = self.t + self._lognorm(self.F["rest_med_s"], self.F["rest_sigma"])
                    self.hard = self.rng.random() < float(self.F.get("q_hard", 0.0))
                    # behind the sofa, under the stools: hidden for this whole rest, sometimes
                    self.hidden = bool(self._inside("occluders", self.px, self.py)) and \
                        self.rng.random() < float(self.F.get("p_hidden", 0.0))
                self.vx = self.vy = 0.0
            else:                                       # back from away: through a doorway (or an edge) to a spot
                ex = self._scene("exits")
                if ex:
                    e = ex[int(self.rng.integers(len(ex)))]
                    self.px, self.py = float(sum(e["pan"]) / 2), float(sum(e["tilt"]) / 2)
                else:
                    self.px = -self.FW / 2 if self.rng.random() < 0.5 else self.XM + self.FW / 2
                self.goal = self._spot()
                sp = self._lognorm(self.F["walk_speed_med"], self.F["walk_speed_sigma"])
                d = max(1.0, math.hypot(self.goal[0] - self.px, self.goal[1] - self.py))
                self.vx, self.vy = sp * (self.goal[0] - self.px) / d, sp * (self.goal[1] - self.py) / d
                self.dog_state, self.state_until = "move", self.t + d / sp
        if self.dog_state == "move":
            self.px += self.vx * dt
            self.py += self.vy * dt

    # ---------------------------------------------------------------- physics
    def advance(self, dt):
        n = max(1, int(round(dt / self.dt)))
        for _ in range(n):
            self._dog_step(self.dt)
            if self.target is not None and self.t >= self.target[2]:
                tx, ty, _ = self.target
                step = self.speed * self.dt
                dx, dy = tx - self.C, ty - self.Y
                dist = math.hypot(dx, dy)
                if dist <= step:
                    self.C, self.Y, self.target = tx, ty, None
                else:
                    self.C += step * dx / dist
                    self.Y += step * dy / dist
                half = self.backlash / 2                    # the picture moves only once the gears take up the play
                if self.C - self.X > half:
                    self.X = self.C - half
                elif self.X - self.C > half:
                    self.X = self.C + half
            if self.t >= self.next_owner:                  # someone turns the camera with the joystick
                p = self._presets[int(self.rng.integers(len(self._presets)))]
                self.target = (float(p["x"]), float(p["y"]), self.t + 0.1)
                self.moved = self.moved_by_owner = self.t
                om = float(self.F.get("owner_moves_per_h") or 0.0)
                self.next_owner = self.t + self.rng.exponential(3600 / om)
            if self.t >= self.next_people:
                self.people_motion = (self.t, float(self.rng.choice([-0.45, 0.45])), float(self.rng.uniform(-0.3, 0.3)))
                self.next_people = self.t + self.rng.exponential(3600 / max(self.F["people_motion_per_h"], 1e-6))
            self.t += self.dt
            self.cam_hist.append((self.t, self.X, self.Y))
            self.dog_hist.append((self.t, self.px, self.py, self.dog_state == "move", self.dog_state, self.hidden))
            self.truth.append((self.t, self.dt, self.in_view(), self.dog_state != "away"))
        if len(self.cam_hist) > 400:
            self.cam_hist = self.cam_hist[-200:]
            self.dog_hist = self.dog_hist[-200:]

    def sleep(self, s):
        self.advance(max(self.dt, s))

    # ---------------------------------------------------------------- what the camera sees
    def offset(self, X=None, Y=None, px=None, py=None):
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        px = self.px if px is None else px
        py = self.py if py is None else py
        return (X - px) / self.FW, (Y - py) / self.FH       # +f = she is right of centre (view right = lower pan)

    def in_view(self, X=None, Y=None, px=None, py=None):
        if px is None and (self.dog_state == "away" or self.hidden):
            return False
        f, g = self.offset(X, Y, px, py)
        return abs(f) < 0.45 and abs(g) < 0.45

    def _past(self, hist, lag):
        tt = self.t - lag
        for h in reversed(hist):
            if h[0] <= tt:
                return h
        return hist[0]

    def frame(self):
        """What a frame arriving now shows: the scene `lag` seconds ago. Returns (box or None, moving, weak boxes)."""
        lag = self.F["lag_s"]
        _, X, Y = self._past(self.cam_hist, lag)
        _, px, py, mv, st, hid = self._past(self.dog_hist, lag)    # camera AND pet at the same exposure time
        weak = []
        if st != "away" and not hid and self.in_view(X, Y, px, py):
            f, g = self.offset(X, Y, px, py)
            p = self.P["detect_persist"]["moving" if mv else "still"]["p"]
            if not mv and getattr(self, "hard", False):
                p = float(self.F.get("p_hard", p))
            box = [0.5 + f - 0.08, 0.5 + g - 0.1, 0.5 + f + 0.08, 0.5 + g + 0.1]
            box = [v + float(self.rng_sense.uniform(-0.012, 0.012)) for v in box]
            if self.rng_sense.random() < p:
                return box, mv, weak
            if self.rng_sense.random() < 0.8:
                weak.append(box)
            return None, False, weak
        if self.rng_sense.random() < self.F["p_false_accept"]:
            c = float(self.rng_sense.uniform(0.2, 0.8)), float(self.rng_sense.uniform(0.3, 0.8))
            return [c[0] - 0.06, c[1] - 0.08, c[0] + 0.06, c[1] + 0.08], False, weak
        return None, False, weak

    def motion(self):
        """The frame reader's motion centroid, if the picture changed lately: her, at or near the edge, or people."""
        lag = self.F["lag_s"]
        _, X, Y = self._past(self.cam_hist, lag)
        _, px, py, mv, st, _hid = self._past(self.dog_hist, lag)
        if mv and st != "away":
            f, g = self.offset(X, Y, px, py)
            if abs(f) < 0.6 and abs(g) < 0.6:
                return (self.t - 0.2, max(-0.49, min(0.49, f)), max(-0.49, min(0.49, g)))
        if self.people_motion and self.t - self.people_motion[0] < 1.0:
            return self.people_motion
        return None

    # ---------------------------------------------------------------- camera API (what server/tracker.py calls)
    def _start(self, x, y, kind):
        self.wait_idle()                                   # the app waits for the move in progress, then moves
        x, y = max(0.0, min(self.XM, x)), max(40.0, min(self.YM - 40.0, y))
        self.moves.append((self.t, kind, abs(x - self.C) + abs(y - self.Y)))
        self.target = (x, y, self.t + 0.1)                 # ~0.1 s from command to first motion
        self.moved = self.t

    def position(self): return round(self.C), round(self.Y)        # the counter, like motors -j
    def limits(self): return self.XM, self.YM
    def frame_steps(self): return self.FW, self.FH
    def presets(self): return self._presets
    def home(self): return 1850, 500
    def moved_at(self): return self.moved_by_owner     # the last joystick move (pauses Follow, USER_PAUSE)
    def busy(self): return self.target is not None
    def stop(self): self.target = None
    def goto(self, x, y): self._start(x, y, "goto")
    def aim(self, fx, fy): self._start(self.C - fx * self.FW, self.Y - fy * self.FH, "aim")   # the app aims from the counter

    def wait_idle(self):
        n = 0
        while self.busy() and n < 2000:
            self.advance(self.dt)
            n += 1
        return self.position()

    FRAME_DT = 0.2                                          # live frames arrive at 5 fps (server/watch.py FRAME_FPS)

    def _next_arrival(self, t):
        return math.ceil(t / self.FRAME_DT + 1e-9) * self.FRAME_DT

    def find_pet(self, after):
        """A settled look, with the watcher's rule (server/watch.py find_pet): a frame TAKEN after `after` (it arrives
        lag later, on the 5 fps grid) and not one already judged; plus the detector's ~0.05 s. Review 2026-11-04: this
        used to return, 0.4 s on, whatever delayed frame was there, i.e. one exposed BEFORE the requested time."""
        t = self._next_arrival(max(self.t, after + self.F["lag_s"], getattr(self, "last_judged", -1.0) + 1e-6))
        self.sleep(max(0.0, t - self.t) + 0.05)
        self.last_judged = t
        return self.frame()[0]

    def find_now(self, since):
        """Mid-pan: the newest frame if one arrived after `since`, else nothing new (no wait)."""
        newest = math.floor(self.t / self.FRAME_DT + 1e-9) * self.FRAME_DT
        if newest <= since or newest <= getattr(self, "last_judged", -1.0):
            return None
        self.last_judged = newest
        box = self.frame()[0]
        if box is not None:
            box = _Seen(box)
            box.captured = newest - self.F["lag_s"]     # the frame shows the scene lag_s before it arrived
        return box
