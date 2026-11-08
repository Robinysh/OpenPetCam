#!/usr/bin/env python3
"""How Follow behaves on the real camera, from the watcher's own records (read-only).

Per hour of watched time it counts tracking aims, aims that swing back the other way within 15 s (a reversal: the
camera crossing her and back), and the share of cycles in which she was verified. "Active" hours are the ones in
which Follow aimed at least MIN_AIMS times: resting hours say nothing about following.

    python3 tools/follow_live.py EXPORT.json [--since UNIX] [--until UNIX]    # an export from tools/export_logs.py
Compare a change by running it on the hours before and after it (same hours of the day where possible).
"""
import argparse
import json
import re
import statistics

MIN_AIMS = 6
AIM = re.compile(r"^track[^:]*: aim \(([-+.\d]+), ([-+.\d]+)\)")


def rows(path, since=None, until=None):
    d = json.load(open(path))
    obs = d["obs"]
    track = d["track"]
    keep = lambda ts: (since is None or ts >= since) and (until is None or ts < until)   # noqa: E731
    o = [r for r in obs if keep(r[0] if isinstance(r, list) else r["ts"])]
    t = [r for r in track if keep(r[0] if isinstance(r, list) else r["ts"])]
    return o, t


def field(r, i, name):
    return r[i] if isinstance(r, list) else r.get(name)


def metrics(obs, track):
    hours = {}
    for r in obs:
        ts, score = field(r, 0, "ts"), field(r, 1, "score") or 0
        h = hours.setdefault(int(ts // 3600), {"cycles": 0, "seen": 0, "aims": [], "moves": 0})
        h["cycles"] += 1
        h["seen"] += score >= 0.5
    for r in track:
        ts, act = field(r, 0, "ts"), field(r, 3, "act") or ""
        h = hours.get(int(ts // 3600))
        if h is None:
            continue
        m = AIM.match(act)
        if m:
            h["aims"].append((ts, float(m.group(1)), float(m.group(2))))
    out = []
    for k, h in sorted(hours.items()):
        a = h["aims"]
        rev = sum(1 for (t0, x0, _), (t1, x1, _) in zip(a, a[1:]) if t1 - t0 <= 15 and x0 * x1 < 0 and min(abs(x0), abs(x1)) >= 0.1)
        travel = sum(abs(x) for _, x, _ in a)
        out.append({"hour": k, "cycles": h["cycles"], "seen_share": round(h["seen"] / max(h["cycles"], 1), 3),
                    "aims": len(a), "reversals": rev, "travel_frames": round(travel, 2)})
    return out


def summary(hours):
    act = [h for h in hours if h["aims"] >= MIN_AIMS]
    if not act:
        return {"active_hours": 0}
    med = lambda k: round(statistics.median(h[k] for h in act), 3)     # noqa: E731
    return {"active_hours": len(act), "aims_per_active_h": med("aims"), "reversals_per_active_h": med("reversals"),
            "reversal_share": round(sum(h["reversals"] for h in act) / max(sum(h["aims"] for h in act), 1), 3),
            "travel_frames_per_active_h": med("travel_frames"), "seen_share_active_h": med("seen_share")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--since", type=float)
    ap.add_argument("--until", type=float)
    a = ap.parse_args()
    h = metrics(*rows(a.export, a.since, a.until))
    print(json.dumps(summary(h), indent=1))
