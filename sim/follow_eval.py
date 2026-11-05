#!/usr/bin/env python3
"""Follow variants compared in the room-model simulator, on ground truth only a simulator has.

The real controller (server/tracker.py) runs in HomeRoom (sim/home.py) the way the watcher runs it (gate1.run_watcher),
once per variant on the SAME seeds and start hours (paired): today's Follow (picture coordinates) and the room-aware
Follow (tracker.METRIC_FOLLOW with roomfollow.RoomFollow from the room bundle). Scored on what the simulator knows and
the logs cannot: the true share of time she is in the camera's view while home, motor moves per hour, and the logged
losses per home hour. Differences are paired by seed with a bootstrap interval.
    python3 sim/follow_eval.py [world] [export] [hours] [seeds] [params.json]   -> eval/rl/follow_eval.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.append(os.path.join(ROOT, "tools"))
import gate1  # noqa: E402
from home import HomeData, HomeRoom  # noqa: E402
from room import Room, load_params  # noqa: E402


def run(world, export, hours=4.0, seeds=12, params_path=None, walks=None):
    from zoneinfo import ZoneInfo
    import roomfollow
    try:
        from siteconf import SITE
        tz = ZoneInfo(SITE.get("TZ", "UTC"))
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    P = load_params(params_path) if params_path else load_params()
    free = P.get("free") or {}
    Room.HARD_CORRELATED = True
    data = HomeData.from_world(world, export, tz=tz, walks=walks)
    doc = json.load(open(os.path.join(ROOT, "models", "room", "room.json")))
    rf = roomfollow.RoomFollow.load(doc, os.path.join(ROOT, "models", "room", doc["textures"]["range"]))
    hours_of_day = [s["hour"] for s in data.stays]
    out = {}
    for name, metric in (("today", False), ("room_aware", True)):
        T = gate1.load_tracker()
        T.ROOM_FOLLOW, T.METRIC_FOLLOW = (rf, True) if metric else (None, False)
        rows = []
        for s in range(seeds):
            start = hours_of_day[s * 7919 % len(hours_of_day)]
            room = HomeRoom(seed=1000 + s, params=P, free=free, home=data, start_hour=start)
            obs, acts = gate1.run_watcher(room, T, hours)
            m, tr = gate1.metrics(obs, acts), gate1.truth(room)
            rows.append({"seed": s, "true_in_view": tr["true_in_view_share_while_home"], "moves_per_hour": tr["moves_per_hour"],
                         "losses_per_home_hour": m["losses_per_home_hour"], "logged_in_view": m["in_view_share_while_home"]})
        out[name] = rows
        print(name, {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0] if k != "seed"}, flush=True)
    rng = np.random.default_rng(0)
    diff = {}
    for k in ("true_in_view", "moves_per_hour", "losses_per_home_hour"):
        d = np.array([b[k] - a[k] for a, b in zip(out["today"], out["room_aware"])])
        bs = [float(np.mean(rng.choice(d, len(d)))) for _ in range(4000)]
        diff[k] = {"mean": round(float(d.mean()), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    res = {"hours": hours, "seeds": seeds, "rows": out, "room_aware_minus_today": diff}
    json.dump(res, open(os.path.join(ROOT, "eval", "rl", "follow_eval.json"), "w"), indent=1)
    print("room-aware minus today:", json.dumps(diff), flush=True)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0] if a else os.path.join(ROOT, "eval", "world"), a[1] if len(a) > 1 else os.path.join(ROOT, "eval", "rl", "rl_export.json"),
        float(a[2]) if len(a) > 2 else 4.0, int(a[3]) if len(a) > 3 else 12, a[4] if len(a) > 4 else None,
        os.path.join(ROOT, "eval", "rl", "walks.json"))


def prediction_test(world, export, walks=None, hours=6, seed0=2000):
    """Where will she be LEAD s later? Pixel predictor (today's tracker), metric predictor (roomfollow), and no
    prediction (her last box), scored against where the simulator knows her body centre really is then."""
    import math
    from zoneinfo import ZoneInfo
    import roomfollow
    P = load_params(); free = P.get("free") or {}
    Room.HARD_CORRELATED = True
    data = HomeData.from_world(world, export, tz=ZoneInfo("America/New_York"), walks=walks)
    doc = json.load(open(os.path.join(ROOT, "models", "room", "room.json")))
    rf = roomfollow.RoomFollow.load(doc, os.path.join(ROOT, "models", "room", doc["textures"]["range"]))
    T = gate1.load_tracker(); LEAD = T.LEAD
    rows = []
    for k in range(hours):
        room = HomeRoom(seed=seed0 + k, params=P, free=free, home=data, start_hour=data.stays[(k * 31) % len(data.stays)]["hour"])
        mt = roomfollow.MetricTrack(rf); hist = []
        end = room.t + 3600
        while room.t < end:
            box, mv, _ = room.frame()
            if box is not None and room.dog_state == "move":
                pan, tilt = room.position()
                cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
                t_cap = room.t - room.F["lag_s"]
                hist.append((t_cap, cx, cy)); hist = [h for h in hist if t_cap - h[0] <= 4]
                mt.add(t_cap, pan, tilt, box)
                fut = (room.px + room.vx * LEAD, room.py + room.vy * LEAD)
                o_true = rf.offset(pan, tilt, rf.body_centre(*fut))
                if o_true and len(hist) >= 3 and len(mt.pts) >= 3:
                    (t0, x0, y0), (t1, x1, y1) = hist[0], hist[-1]; dt = t1 - t0
                    vxp, vyp = ((x1 - x0) / dt, (y1 - y0) / dt) if dt >= 0.8 else (0, 0)
                    lead = LEAD + (room.t - t_cap)
                    pix = (hist[-1][1] + vxp * lead, hist[-1][2] + vyp * lead)
                    a = mt.aim(pan, tilt, room.t + LEAD)
                    if a:
                        rows.append({"pixel": math.hypot(pix[0] - o_true[0], pix[1] - o_true[1]),
                                     "metric": math.hypot(a[0] - o_true[0], a[1] - o_true[1]),
                                     "none": math.hypot(cx - o_true[0], cy - o_true[1])})
            room.advance(1.0)
    json.dump({"lead_s": LEAD, "rows": rows}, open(os.path.join(ROOT, "eval", "rl", "prediction_test.json"), "w"))
    return rows
