#!/usr/bin/env python3
"""Gate 1: run the REAL controller (server/tracker.py, unchanged) in the simulated room the way the watcher runs it,
log what the watcher would have logged, and compare those logged statistics with the real ones (sim/params.json
real_metrics). Nothing may be tuned or trained on a simulator that fails this.

    python3 sim/gate1.py [hours] [seeds]          prints sim vs real, per metric
"""
import collections
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from room import Room, load_params  # noqa: E402

SEEN, LOSS_GAP, LEAVE_AFTER = 0.30, 6.0, 180.0
DETECT_EVERY, DETECT_STILL = 8.0, 30.0   # server/watch.py as the fitted logs were recorded (DETECT_STILL is 12 since
                                         # 2026-11-04; switch here and re-fit once a week of logs at 12 s exists)


def load_tracker(path=os.path.join(HERE, "..", "server", "tracker.py"), rev=None):
    """The controller under test: today's server/tracker.py, or the exact version a past log was made with (rev =
    the git commit in track.policy's "arm@version"), so a replay runs the code that actually ran."""
    if rev:
        import subprocess
        import tempfile
        src = subprocess.run(["git", "show", f"{rev}:server/tracker.py"], capture_output=True, text=True, check=True,
                             cwd=os.path.join(HERE, "..")).stdout
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        f.write(src)
        f.close()
        path = f.name
    spec = importlib.util.spec_from_file_location("tracker_under_test", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_watcher(room, T, hours, overrides=None):
    """The watcher loop, as server/watch.py runs it, around one Tracker. Returns (obs rows, follow actions)."""
    for k, v in (overrides or {}).items():
        setattr(T, k, v)
    tr = T.Tracker(room, room.find_pet, clock=lambda: room.t, log=lambda m: None, sleep=room.sleep, find_now=room.find_now)
    obs, acts = [], []
    last_seen, present = -1e9, False
    last_detect, last_motion_t, result = -1e9, -1e9, (None, False, [])
    end = room.t + hours * 3600
    while room.t < end:
        motion = room.motion()
        if motion:
            last_motion_t = room.t
        # server/watch.py: the detector runs on motion, or every DETECT_EVERY s while lively, else every DETECT_STILL s;
        # in between, each cycle repeats the last result (a stale miss stays a miss until the next run)
        lively = room.t - last_motion_t < 60 or tr.searching or (room.t - last_seen < 20)
        fresh = bool(motion) or room.t - last_detect >= (DETECT_EVERY if lively else DETECT_STILL)
        if fresh:
            result, last_detect = room.frame(), room.t
        box, moving, weak = result
        weak = weak if fresh else []
        if box is not None:
            last_seen, present = room.t, True
        elif room.t - last_seen > LEAVE_AFTER:
            present = False
        obs.append((room.t, 1.0 if box else 0.0, int(present), int(bool(moving and box))))
        t_before = room.t
        a = tr.step(box, moving=bool(moving and box), sound=False, motion=motion, weak=weak or None, box_t=last_detect)
        if a:
            acts.append((t_before, a.split(":")[0].split(" ")[0], a))
        # next cycle: ~1 s while motion wakes the loop, else the 3 s period (server/watch.py MIN_CYCLE, PERIOD)
        period = 1.0 if motion else 3.0
        room.advance(max(0.05, period - (room.t - t_before)))
    return obs, acts


def metrics(obs, acts):
    losses, last, lost_at = [], None, None
    for ts, score, *_ in obs:
        if score >= SEEN:
            if lost_at is not None:
                losses.append(ts - lost_at)
                lost_at = None
            last = ts
        elif last is not None and lost_at is None and ts - last >= LOSS_GAP:
            lost_at = last
    home_h = sum(min(b[0] - a[0], 30) for a, b in zip(obs, obs[1:]) if a[2]) / 3600
    seen_h = sum(min(b[0] - a[0], 30) for a, b in zip(obs, obs[1:]) if a[1] >= SEEN) / 3600
    span_h = (obs[-1][0] - obs[0][0]) / 3600 if obs else 1
    c = collections.Counter(x[1] for x in acts)
    ends = [x[2] for x in acts if len(x) > 2 and x[2].startswith("search (")]
    found = [e for e in ends if "found her" in e]
    within = lambda s: round(sum(x <= s for x in losses) / max(len(losses), 1), 3)   # noqa: E731
    return {"in_view_share_while_home": round(seen_h / max(home_h, 1e-9), 3),
            "losses_per_home_hour": round(len(losses) / max(home_h, 1e-9), 2),
            "seen_again_within_s": {s: within(s) for s in (10, 30, 60, 600)},
            "follow_actions_per_hour": {k: round(v / span_h, 2) for k, v in c.items()},
            "home_share": round(home_h / max(span_h, 1e-9), 3),
            "search_found_share": round(len(found) / len(ends), 3) if ends else None}


def truth(room):
    home = sum(dt for _, dt, _, h in room.truth if h)
    seen = sum(dt for _, dt, v, h in room.truth if h and v)
    moves = len(room.moves)
    return {"true_in_view_share_while_home": round(seen / max(home, 1e-9), 3),
            "moves_per_hour": round(moves / max(room.t / 3600, 1e-9), 2)}


ROOM = Room      # the simulator class to run: sim/room.py's schematic Room, or sim/home.py's HomeRoom (home_gate.py)


def simulate(hours=6, seeds=4, free=None, overrides=None, params=None, seed0=0):
    T = load_tracker()
    rows = []
    for s in range(seed0, seed0 + seeds):
        room = ROOM(seed=s, free=free, params=params)
        obs, acts = run_watcher(room, T, hours, overrides)
        rows.append((metrics(obs, acts), truth(room)))
    return rows


def compare(rows, real):
    out = {}
    keys = [("in_view_share_while_home",), ("losses_per_home_hour",), ("seen_again_within_s", 30), ("seen_again_within_s", 60),
            ("follow_actions_per_hour", "chase"), ("follow_actions_per_hour", "search"), ("follow_actions_per_hour", "track")]
    for k in keys:
        get = lambda m: (m.get(k[0], {}) or {}).get(k[1], 0.0) if len(k) == 2 else m.get(k[0])   # noqa: E731
        sims = sorted(get(m) for m, _ in rows)
        r = (real.get(k[0], {}) or {}).get(str(k[1]), (real.get(k[0], {}) or {}).get(k[1])) if len(k) == 2 else real.get(k[0])
        lo, hi = sims[0], sims[-1]
        out["/".join(map(str, k))] = {"real": r, "sim_min": lo, "sim_max": hi, "sim_mean": round(sum(sims) / len(sims), 3)}
    return out


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    P = load_params()
    free = None if "--defaults" in sys.argv else P.get("free")      # the calibrated room unless asked for the defaults
    if len(sys.argv) > 3 and sys.argv[3].startswith("lag="):
        free = dict(free or {}, lag_s=float(sys.argv[3][4:]))
    rows = simulate(hours, seeds, free=free, params=P)
    real = P["real_metrics"]
    for k, v in compare(rows, real).items():
        print(f"{k:<36} real {v['real']!s:<7} sim {v['sim_mean']:<7} [{v['sim_min']}, {v['sim_max']}]")
    print("truth (sim only):", [t for _, t in rows])
