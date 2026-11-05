"""The truthful simulator: the same camera physics and watcher interface as sim/room.py, in the real room.

What changes against the schematic Room (sim/room.py), and where each part comes from:
  where she is    metres on the measured floor, not pan/tilt steps. Her stays (where she lies, how long, by hour of
                  day), her walks (real walking speeds between stays), her absences (how long, where she leaves and
                  comes back) are all resampled from her real detections on TRAINING days only (no fitted habits).
  what the camera sees   her body (a box, standing or lying) projected through the calibrated camera at the picture's
                  real pose, with the mount's measured drift for that time of day; a body point is hidden when the
                  measured room (the fused surface, blind spots filled by sound) has something nearer in its
                  direction. Hidden is geometric, not a probability.
  the box         the bounding box of her visible body points in the frame, so a dog cut off by the frame edge or
                  half behind the sofa gives the box the detector would see.
Unchanged from Room: the motor (speed, start, backlash), frame timing and lag, the watcher's API, people motion,
owner moves, and the detection model's free parameters (detect persistence, hard poses), which are calibrated
against the training days' logged statistics exactly as for Room.

    from home import HomeData, HomeRoom
    data = HomeData.from_world("eval/world", export="eval/rl/rl_export.json", train_days={...})
"""
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from room import Room  # noqa: E402

STAY_R, STAY_GAP, WALK_GAP, AWAY_GAP = 0.5, 300.0, 300.0, 1800.0
BODY = {"stand": (0.80, 0.30, 0.55), "lie": (0.90, 0.45, 0.28)}      # length, width, height (m): a golden retriever
VISIBLE_SHARE = 0.4                                                   # of body points for "in view"


def day_of(ts, tz):
    return datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d")


class HomeData:
    """The measured room and her real behaviour, from the room model and the training days' records."""

    def __init__(self, calib, R, floor_h, drift, rows, tz, gait=None):
        from world.camera import Camera
        self.cam = Camera(calib)
        self.R, self.H, self.W = R, R.shape[0], R.shape[1]
        self.floor_h, self.drift, self.tz = floor_h, drift, tz
        self.rows = sorted(rows)
        self.gait = list(gait or [])                    # walking speeds (m/s) measured from continuous video (walks.py)
        self._segment()

    @classmethod
    def from_world(cls, world, export, train_days=None, tz=None, walks=None):
        from zoneinfo import ZoneInfo
        sys.path.insert(0, os.path.join(HERE, "..", "server"))
        import roomcam
        tz = tz or ZoneInfo("UTC")
        model = os.path.join(world, "model")
        m = json.load(open(os.path.join(model, "manifest.json")))
        fs = next(x for x in m["layers"] if x["id"] == "fused.surface")["data"]
        R = np.fromfile(os.path.join(model, fs.get("filled_file") or fs["file"]), np.float32).reshape(fs["height"], fs["width"])
        calib = json.load(open(os.path.join(world, "calib.json")))
        reg = os.path.join(world, "raw", "registration.json")
        drift = (json.load(open(reg)).get("smooth") or []) if os.path.exists(reg) else []
        floor_h = m["room"]["floor_height_m"]
        # her positions from the export, with the server's own rule (roomapi), on training days only
        d = json.load(open(export))
        con = sqlite3.connect(":memory:")
        con.executescript("CREATE TABLE obs(ts REAL, score REAL, person REAL, box TEXT, motion REAL, present INT, moving INT);"
                          "CREATE TABLE track(ts REAL, pan INT, tilt INT, act TEXT, weak INT);")
        keep = (lambda ts: day_of(ts, tz) in train_days) if train_days else (lambda ts: True)
        con.executemany("INSERT INTO obs VALUES(?,?,?,?,?,?,?)", [r for r in d["obs"] if r[3] and keep(r[0])])
        con.executemany("INSERT INTO track VALUES(?,?,?,?,?)", [r for r in d["track"] if keep(r[0])])
        rc = roomcam.Camera({k: calib[k] for k in ("focal_px", "k1", "k2", "deg_per_step_pan", "deg_per_step_tilt", "level_tilt", "roll_deg")})
        rows = []
        tt, pp = _poses(con)
        last_box, last_ts = None, None
        for ts, box, present, moving in con.execute("SELECT ts, box, present, moving FROM obs ORDER BY ts"):
            if not present:
                last_box = None
                continue
            if box == last_box:
                if rows and ts - last_ts < 30:
                    rows[-1][4] = ts - rows[-1][0] + 1.5
                last_ts = ts
                continue
            last_box, last_ts = box, ts
            pose = _pose_at(tt, pp, ts)
            if pose is None:
                continue
            b = json.loads(box)
            off = _drift_at(drift, ts)
            p = roomcam.floor_point(rc, pose[0], pose[1], b, floor_h, off)
            if p is not None and b[3] <= 0.97:                # boxes cut off at the bottom edge sit too far: dropped
                rows.append([ts, p[0], p[1], int(moving or 0), 1.5])
        gait = []
        if walks and os.path.exists(walks):             # her gait from video: 1 s windows above the box-jitter floor
            for it in json.load(open(walks))["items"]:
                if not it["moving"]:
                    continue
                P, j = it["points"], 0
                for k in range(len(P)):
                    while P[k][0] - P[j][0] > 1.0:
                        j += 1
                    if P[k][0] - P[j][0] >= 0.8:
                        v = math.hypot(P[k][1] - P[j][1], P[k][2] - P[j][2]) / (P[k][0] - P[j][0])
                        if v > 0.3:
                            gait.append(v)
        return cls(calib, R, floor_h, drift, rows, tz, gait)

    # ---- her behaviour, segmented from the detections ----
    def _segment(self):
        stays, cur = [], None
        for t, x, z, mv, dw in self.rows:
            if cur and math.hypot(x - cur["x"], z - cur["z"]) < STAY_R and t - cur["end"] < STAY_GAP:
                n = cur["n"]
                cur["x"], cur["z"] = (cur["x"] * n + x) / (n + 1), (cur["z"] * n + z) / (n + 1)
                cur["n"], cur["end"] = n + 1, max(cur["end"], t + dw)
            else:
                if cur:
                    stays.append(cur)
                cur = {"start": t, "end": t + dw, "x": x, "z": z, "n": 1}
        if cur:
            stays.append(cur)
        self.stays = [s for s in stays if s["end"] - s["start"] >= 5]
        self.walks, self.absences, self.after = [], [], []
        for a, b in zip(self.stays, self.stays[1:]):
            gap = b["start"] - a["end"]
            hr = self.hour(a["end"])
            if gap <= WALK_GAP:
                self.after.append((hr, "walk"))
            elif gap >= AWAY_GAP:
                self.absences.append({"hour": hr, "dur": gap, "out": (a["x"], a["z"]), "back": (b["x"], b["z"])})
                self.after.append((hr, "away"))
            else:
                self.after.append((hr, "walk"))                 # unseen for a while but home: a walk to her next spot
        for s in self.stays:
            s["hour"] = self.hour(s["start"])
        # small moves while she stays: the share of her detections inside stays that are 'moving'
        inside = [r for r in self.rows if any(st["start"] <= r[0] <= st["end"] for st in self.stays)] if len(self.stays) < 2000 else self.rows
        self.fidget_share = float(np.mean([r[3] for r in inside])) if inside else 0.0
        # walking speed: consecutive MOVING detections a few seconds apart (the gap between stays includes unseen time)
        for (t1, x1, z1, m1, _), (t2, x2, z2, m2, _) in zip(self.rows, self.rows[1:]):
            if m1 and m2 and 0.5 <= t2 - t1 <= 6:
                sp = math.hypot(x2 - x1, z2 - z1) / (t2 - t1)
                if 0.05 <= sp <= 3.0:
                    self.walks.append(sp)

    def hour(self, ts):
        return datetime.fromtimestamp(ts, self.tz).hour

    def near_hour(self, items, hour, key=lambda s: s["hour"], width=1):
        pool = [s for s in items if min((key(s) - hour) % 24, (hour - key(s)) % 24) <= width]
        return pool or list(items)

    def summary(self):
        return {"detections": len(self.rows), "stays": len(self.stays), "walks": len(self.walks),
                "absences": len(self.absences), "walk_speed_median_m_s": round(float(np.median(self.walks)), 2) if self.walks else None,
                "stay_median_s": round(float(np.median([s["end"] - s["start"] for s in self.stays])), 1) if self.stays else None,
                "away_median_s": round(float(np.median([a["dur"] for a in self.absences])), 1) if self.absences else None}

    def range_at(self, d):
        az = np.degrees(np.arctan2(d[..., 0], d[..., 2])); el = np.degrees(np.arcsin(np.clip(d[..., 1], -1, 1)))
        i = ((az + 180) / 360 * self.W).astype(int) % self.W
        j = np.clip(((90 - el) / 180 * self.H).astype(int), 0, self.H - 1)
        return self.R[j, i]


def _poses(con):
    rows = con.execute("SELECT ts, pan, tilt FROM track ORDER BY ts").fetchall()
    return [r[0] for r in rows], [(r[1], r[2]) for r in rows]


def _pose_at(tt, pp, ts, gap=5.0):
    import bisect
    if not tt:
        return None
    k = bisect.bisect_left(tt, ts)
    cand = [i for i in (k - 1, k) if 0 <= i < len(tt)]
    i = min(cand, key=lambda i: abs(tt[i] - ts))
    if abs(tt[i] - ts) > gap or pp[max(i - 1, 0)] != pp[min(i + 1, len(pp) - 1)]:
        return None
    return pp[i]


def _drift_at(smooth, ts):
    if not smooth:
        return (0.0, 0.0, 0.0)
    k = min(range(len(smooth)), key=lambda i: abs(smooth[i][0] - ts))
    return (smooth[k][1], smooth[k][2], 0.0) if abs(smooth[k][0] - ts) <= 7200 else (0.0, 0.0, 0.0)


class HomeRoom(Room):
    """Room's camera, timing and watcher API; her body and the camera's view in the measured room."""

    # her state for everything outside her behaviour model (Room records it per step): a small move inside a stay
    # counts as moving, so the detector and the motion reader see it like a walk
    @property
    def dog_state(self):
        st = getattr(self, "_state", "rest")
        return "move" if st == "rest" and getattr(self, "_fidgeting", False) else st

    @dog_state.setter
    def dog_state(self, v):
        self._state = v

    def __init__(self, seed=0, params=None, free=None, presets=None, home=None, start_hour=None):
        super().__init__(seed=seed, params=params, free=free, presets=presets)
        self.home = home
        self.rng_dog = np.random.default_rng([seed, 31337])
        hours = [s["hour"] for s in home.stays] or [12]
        self.base_hour = float(start_hour if start_hour is not None else self.rng_dog.choice(hours))
        self.base_drift = home.drift[int(self.rng_dog.integers(len(home.drift)))][1:3] if home.drift else (0.0, 0.0)
        self.heading = float(self.rng_dog.uniform(0, 2 * math.pi))
        s = self._pick_stay()
        self.px, self.py = s["x"], s["z"]
        self._state, self.state_until = "rest", self.t + self._stay_len()
        self.hidden = False
        self.dog_hist = [(0.0, self.px, self.py, False, self.dog_state, False)]
        self._vis_cache = {}

    # ---- clock and drift ----
    def hour_now(self):
        return int((self.base_hour + self.t / 3600) % 24)

    def drift_now(self):
        return (float(self.base_drift[0]), float(self.base_drift[1]), 0.0)

    # ---- her behaviour (resampled from training days) ----
    def _pick_stay(self):
        pool = self.home.near_hour(self.home.stays, self.hour_now())
        return pool[int(self.rng_dog.integers(len(pool)))]

    def _stay_len(self):
        pool = self.home.near_hour(self.home.stays, self.hour_now())
        s = pool[int(self.rng_dog.integers(len(pool)))]
        return max(5.0, s["end"] - s["start"])

    def _speed(self):
        if self.home.gait:                              # measured from video (walks.py)
            return float(self.rng_dog.choice(self.home.gait))
        # without video her walking speed is not observable (moving detections are mostly shifting in place)
        return max(0.1, float(self.F.get("walk_speed_ms", 0.8)) * math.exp(self.rng_dog.normal(0, float(self.F.get("walk_speed_sigma", 0.5)))))

    FIDGET_S = 2.0                                       # length of one small move (assumed; its rate is measured)

    def _dog_step(self, dt):
        if self._state == "rest" and self.t < self.state_until and getattr(self, "fidget_until", 0) <= self.t:
            # while she stays she shifts and repositions, at her measured rate (share of moving detections in stays)
            if self.rng_dog.random() < self.home.fidget_share * dt / self.FIDGET_S:
                # 'moving' in her records is motion inside her box (head, shifting), not a walk: on video, still tracks
                # move under 0.09 m/s (90th percentile of 1 s windows), so her body barely moves during it
                a = float(self.rng_dog.uniform(0, 2 * math.pi)); sp = 0.05
                self.fidget = (sp * math.sin(a), sp * math.cos(a))
                self.fidget_until = self.t + self.FIDGET_S
                self.heading = a
                if getattr(self, "hard", False):
                    if self.HARD_CORRELATED:   # drawn only in this mode: the original model's random stream is unchanged
                        self.hard_seen = self.rng.random() < float(self.F.get("p_hard", 0.3))
        if getattr(self, "fidget_until", 0) > self.t and self._state == "rest":
            self.px += self.fidget[0] * dt; self.py += self.fidget[1] * dt
            self._fidgeting = True
        else:
            self._fidgeting = False
        if self.t >= self.state_until:
            if self._state == "rest":
                after = self.home.near_hour(self.home.after, self.hour_now(), key=lambda a: a[0])
                kind = after[int(self.rng_dog.integers(len(after)))][1] if after else "walk"
                if kind == "away" and self.home.absences:
                    pool = self.home.near_hour(self.home.absences, self.hour_now())
                    self.away = pool[int(self.rng_dog.integers(len(pool)))]
                    goal = self.away["out"]
                else:
                    self.away = None
                    s = self._pick_stay()
                    goal = (s["x"] + float(self.rng_dog.normal(0, 0.15)), s["z"] + float(self.rng_dog.normal(0, 0.15)))
                # her walking speed is NOT observable (a walk leaves the frame within a cycle; moving detections are
                # mostly shifting in place, median 0.15 m/s): calibrated, as in Room, here in metres per second
                sp = self._speed()
                d = max(0.05, math.hypot(goal[0] - self.px, goal[1] - self.py))
                self.vx, self.vy = sp * (goal[0] - self.px) / d, sp * (goal[1] - self.py) / d
                self.heading = math.atan2(self.vx, self.vy)
                self._state, self.state_until = "move", self.t + d / sp
            elif self._state == "move":
                self.vx = self.vy = 0.0
                if getattr(self, "away", None):
                    self._state, self.state_until = "away", self.t + float(self.away["dur"])
                else:
                    self._state, self.state_until = "rest", self.t + self._stay_len()
                    self.heading = float(self.rng_dog.uniform(0, 2 * math.pi))
                    self.hard = self.rng.random() < float(self.F.get("q_hard", 0.0))
                    # correlated misses (HARD_CORRELATED): a hard pose is either seen or not for as long as it lasts
                    if self.HARD_CORRELATED:   # drawn only in this mode: the original model's random stream is unchanged
                        self.hard_seen = self.rng.random() < float(self.F.get("p_hard", 0.3))
            else:                                               # back: she reappears where she came back on that day
                self.px, self.py = self.away["back"]
                self.away = None
                self._state, self.state_until = "rest", self.t + self._stay_len()
        if self._state == "move":
            self.px += self.vx * dt
            self.py += self.vy * dt

    # ---- what the camera sees ----
    def _body(self, x, z, state):
        L, Wd, Hh = BODY["stand" if state == "move" else "lie"]
        c, s = math.cos(self.heading), math.sin(self.heading)
        pts = []
        for a in (-0.5, 0.0, 0.5):
            for b in (-0.5, 0.5):
                for hgt in (0.15, 0.85):
                    lx, lz = a * L, b * Wd
                    pts.append((x + lx * s + lz * c, -self.home.floor_h + hgt * Hh, z + lx * c - lz * s))
        return np.array(pts)

    def _view(self, X, Y, px, py, state):
        key = (round(X), round(Y), round(px, 2), round(py, 2), state, round(self.heading, 1))
        if key in self._vis_cache:
            return self._vis_cache[key]
        P = self._body(px, py, state)
        r = np.linalg.norm(P, axis=1)
        d = P / r[:, None]
        u, v, ok = self.home.cam.project(X, Y, d, 1920, 1080, off=self.drift_now())
        rr = self.home.range_at(d)
        clear = ~(np.isfinite(rr) & (rr > 0) & (rr < r * 0.92))
        vis = ok & clear
        cu, cv, cok = self.home.cam.project(X, Y, (np.array([px, -self.home.floor_h + 0.3, py]) /
                                                   np.linalg.norm([px, -self.home.floor_h + 0.3, py]))[None], 1920, 1080, off=self.drift_now())
        centre = (float(cu[0]) / 1920 - 0.5, float(cv[0]) / 1080 - 0.5) if True else None
        res = (vis.mean() >= VISIBLE_SHARE and vis.sum() >= 3,
               [float(u[vis].min()) / 1920, float(v[vis].min()) / 1080, float(u[vis].max()) / 1920, float(v[vis].max()) / 1080] if vis.any() else None,
               centre)
        if len(self._vis_cache) > 20000:
            self._vis_cache.clear()
        self._vis_cache[key] = res
        return res

    def in_view(self, X=None, Y=None, px=None, py=None):
        if px is None and self._state == "away":
            return False
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        px = self.px if px is None else px
        py = self.py if py is None else py
        return self._view(X, Y, px, py, self.dog_state)[0]

    def offset(self, X=None, Y=None, px=None, py=None):
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        px = self.px if px is None else px
        py = self.py if py is None else py
        return self._view(X, Y, px, py, self.dog_state)[2]

    def frame(self):
        lag = self.F["lag_s"]
        _, X, Y = self._past(self.cam_hist, lag)
        _, px, py, mv, st, _hid = self._past(self.dog_hist, lag)
        weak = []
        if st != "away":
            seen, box, _c = self._view(X, Y, px, py, st)
            if seen and box:
                p = self.P["detect_persist"]["moving" if mv else "still"]["p"]
                if not mv and getattr(self, "hard", False):
                    p = (p if getattr(self, "hard_seen", True) else 0.0) if self.HARD_CORRELATED else float(self.F.get("p_hard", p))
                box = [min(1.0, max(0.0, b + float(self.rng_sense.uniform(-0.01, 0.01)))) for b in box]
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
        lag = self.F["lag_s"]
        _, X, Y = self._past(self.cam_hist, lag)
        _, px, py, mv, st, _hid = self._past(self.dog_hist, lag)
        if mv and st != "away":
            f, g = self._view(X, Y, px, py, st)[2]
            if abs(f) < 0.6 and abs(g) < 0.6:
                return (self.t - 0.2, max(-0.49, min(0.49, f)), max(-0.49, min(0.49, g)))
        if self.people_motion and self.t - self.people_motion[0] < 1.0:
            return self.people_motion
        return None


def _main():
    from zoneinfo import ZoneInfo
    sys.path.insert(0, os.path.join(HERE, "..", "tools"))
    try:
        from siteconf import SITE
        tz = ZoneInfo(SITE.get("TZ", "UTC"))
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    a = sys.argv[1:]
    t0 = time.time()
    data = HomeData.from_world(a[0], a[1], tz=tz)
    print(json.dumps(data.summary(), indent=1), f"loaded in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    _main()
