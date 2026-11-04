#!/usr/bin/env python3
"""Fit the Follow simulator to the watcher's own logs (system identification), and measure the real metrics the
simulator must reproduce before anything is trained on it ("gate 1").

    python3 sim/fit_from_logs.py eval/rl/rl_export.json eval/rl/rl_follow.log   ->  sim/params.json

Inputs are the server's obs table (one row per watcher cycle: detection score + box), the track table (camera pan/tilt
and Follow's action per cycle, logged since 2026-11-03) and the follow: lines of the watcher journal. The output holds
only aggregate numbers (distributions, rates); no pictures, names or timestamps leave eval/.

Coordinates: pan 0..3700 and tilt 0..1000 motor steps; one frame is 733 x 424 steps. The view's right is lower pan, and
below the centre is lower tilt (server/app.py aim()), so a box centred at (cx, cy) in the frame puts her at
    pan  - (cx - 0.5) * 733,   tilt - (cy - 0.5) * 424.
"""
import bisect
import json
import math
import os
import re
import sys
from collections import Counter

PAN_FRAME, TILT_FRAME, PAN_MAX, TILT_MAX = 733.0, 424.0, 3700, 1000
SEEN = 0.30            # PRESENT_SCORE in server/watch.py
LOSS_GAP = 6.0         # a loss: seen, then not seen for this long while the watcher kept cycling


def q(xs, ps=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
    xs = sorted(xs)
    return {f"p{int(p * 100)}": round(xs[min(len(xs) - 1, int(p * len(xs)))], 4) for p in ps} if xs else {}


def main(export, follow_log, out):
    d = json.load(open(export))
    obs = d["obs"]                                  # ts, score, person, box, motion, present, moving
    track = d["track"]                              # ts, pan, tilt, act, weak
    t_track = [r[0] for r in track]
    P = {"source": {"obs_rows": len(obs), "track_rows": len(track),
                    "obs_hours": round((obs[-1][0] - obs[0][0]) / 3600, 1) if obs else 0,
                    "track_hours": round((track[-1][0] - track[0][0]) / 3600, 1) if track else 0},
         "geometry": {"pan_max": PAN_MAX, "tilt_max": TILT_MAX, "pan_frame": PAN_FRAME, "tilt_frame": TILT_FRAME}}

    # 1. Sensing: the watcher's cycle interval, by what she was doing
    gaps = {"moving": [], "still": [], "absent": []}
    for a, b in zip(obs, obs[1:]):
        g = b[0] - a[0]
        if g > 30:
            continue
        k = "absent" if a[1] < SEEN else ("moving" if a[6] else "still")
        gaps[k].append(g)
    P["cycle_s"] = {k: q(v) for k, v in gaps.items()}

    # 2. Detection persistence: P(seen next cycle | seen now), by state (a proxy for P(detect | in view))
    persist = {}
    for k, want in (("moving", 1), ("still", 0)):
        n = hit = 0
        for a, b in zip(obs, obs[1:]):
            if a[1] >= SEEN and a[6] == want and b[0] - a[0] <= 3.5:
                n += 1
                hit += b[1] >= SEEN
        persist[k] = {"p": round(hit / max(n, 1), 4), "n": n}
    P["detect_persist"] = persist

    # 3. Her motion in room coordinates: detections joined to the camera pose of the same cycle
    pts = []                                         # (ts, pan_bearing, tilt_bearing, moving)
    for ts, score, _, box, _, _, moving in obs:
        if score < SEEN or not box:
            continue
        i = bisect.bisect_left(t_track, ts - 0.05)
        if i >= len(track) or abs(track[i][0] - ts) > 0.6 or track[i][1] is None:
            continue
        x0, y0, x1, y1 = json.loads(box)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pts.append((ts, track[i][1] - (cx - 0.5) * PAN_FRAME, track[i][2] - (cy - 0.5) * TILT_FRAME, moving))
    speeds, dwell = [], {"move": [], "rest": []}
    run_state, run_t0, prev = None, None, None
    for p in pts:
        if prev and p[0] - prev[0] <= 3.5:
            v = math.hypot(p[1] - prev[1], p[2] - prev[2]) / (p[0] - prev[0])
            if p[3] or prev[3]:
                speeds.append(v)
        st = "move" if p[3] else "rest"
        if run_state != st or (prev and p[0] - prev[0] > 10):
            if run_state and prev:
                dwell[run_state].append(prev[0] - run_t0)
            run_state, run_t0 = st, p[0]
        prev = p
    P["pet"] = {"room_points": len(pts), "speed_steps_s_moving": q(speeds), "dwell_s": {k: q(v) for k, v in dwell.items()}}
    # where she is found: a coarse room map for the search prior (pan bins of half a frame)
    hist = Counter(int(max(0, min(PAN_MAX, p[1])) // (PAN_FRAME / 2)) for p in pts)
    tot = sum(hist.values()) or 1
    P["pet"]["pan_hotspots"] = {str(int(k * PAN_FRAME / 2)): round(v / tot, 4) for k, v in sorted(hist.items())}

    # 4. Real metrics the simulator must reproduce (gate 1)
    losses, last_seen, lost_at = [], None, None
    for ts, score, *_ in obs:
        if score >= SEEN:
            if lost_at is not None:
                losses.append(ts - lost_at)
                lost_at = None
            last_seen = ts
        elif last_seen is not None and lost_at is None and ts - last_seen >= LOSS_GAP:
            lost_at = last_seen
    home_h = sum(min(b[0] - a[0], 30) for a, b in zip(obs, obs[1:]) if a[5]) / 3600
    seen_h = sum(min(b[0] - a[0], 30) for a, b in zip(obs, obs[1:]) if a[1] >= SEEN) / 3600
    acts = Counter()
    span = None
    if os.path.exists(follow_log):
        ts_list = []
        for line in open(follow_log):
            m = re.match(r"([\d.]+) .*follow: (\w+)", line)
            if m:
                acts[m.group(2)] += 1
                ts_list.append(float(m.group(1)))
        # per hour the watcher actually ran (gaps over 30 s excluded), not first-to-last action: a held-out day
        # leaves a hole in the middle that would dilute every rate
        span = sum(min(b[0] - a[0], 30) for a, b in zip(obs, obs[1:])) / 3600 if ts_list else None
    within = lambda s: round(sum(x <= s for x in losses) / max(len(losses), 1), 3)   # noqa: E731
    P["real_metrics"] = {
        "home_hours": round(home_h, 1), "in_view_share_while_home": round(seen_h / max(home_h, 1e-9), 3),
        "losses": len(losses), "losses_per_home_hour": round(len(losses) / max(home_h, 1e-9), 2),
        "seen_again_within_s": {s: within(s) for s in (10, 30, 60, 600)},
        "follow_actions_per_hour": {k: round(v / span, 2) for k, v in acts.items()} if span else {},
    }
    # update only what this script fits; keep measured sections (motor, measured geometry, backlash) and the calibration
    old = json.load(open(out)) if os.path.exists(out) else {}
    geo = dict(P["geometry"], **{k: v for k, v in (old.get("geometry") or {}).items()})
    if P["pet"]["room_points"] < 200 and (old.get("pet") or {}).get("pan_hotspots"):
        # too few sightings with a camera pose (e.g. a day before the pose log existed): keep the known room map
        P["pet"]["pan_hotspots"] = old["pet"]["pan_hotspots"]
        P["pet"]["hotspots_from"] = "previous fit (too few located sightings in these logs)"
    old.update({k: P[k] for k in ("source", "cycle_s", "detect_persist", "pet", "real_metrics")})
    paused = (P["real_metrics"].get("follow_actions_per_hour") or {}).get("paused")
    if paused is not None:          # observable directly: how often people turn the camera by hand
        old.setdefault("free", {})["owner_moves_per_h"] = paused
    old["geometry"] = geo
    old.setdefault("motor", {"measured": "default (run sim/sysid_camera.py)", "speed_steps_s": {"250": 218, "900": 835}})
    P = old
    json.dump(P, open(out, "w"), indent=1)
    print(json.dumps(P, indent=1)[:3000])


if __name__ == "__main__":
    a = sys.argv[1:] or ["eval/rl/rl_export.json", "eval/rl/rl_follow.log"]
    main(a[0], a[1], a[2] if len(a) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json"))
