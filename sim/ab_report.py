#!/usr/bin/env python3
"""Gate 3, live: compare the learned Follow policy with today's constants on the real camera, from the watcher's own
logs. The watcher alternates them in 2-hour blocks (server/watch.py apply_policy) and records which one ran in
track.policy; each block is scored with the same logged metrics gate 1 uses, while she was home.

    python3 sim/ab_report.py [days]
Verdict rule (fixed before the data, 2026-11-04): keep the learned policy only if, over at least 8 home hours per arm,
its logged in-view share is not lower and its moves per home hour are not higher than base; otherwise remove
models/follow_policy.json and redeploy (the watcher then runs today's constants).
"""
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from siteconf import SSH  # noqa: E402

Q = r'''
import sqlite3, json, time, sys
days = float(sys.argv[1]) if len(sys.argv) > 1 else 3
c = sqlite3.connect("/var/lib/petcam/petcam.db")
t0 = time.time() - days * 86400
print(json.dumps({"obs": c.execute("select ts,score,present from obs where ts>? order by ts", (t0,)).fetchall(),
                  "track": c.execute("select ts,act,policy from track where ts>? and policy is not null order by ts", (t0,)).fetchall()}))
'''


def main(days=3):
    raw = subprocess.run(SSH + [f"sudo python3 - {days}"], input=Q, capture_output=True, text=True, timeout=300, check=True).stdout
    d = json.loads(raw)
    arm_at = sorted((t, p.split("@")[0]) for t, _, p in d["track"])   # policy column is "arm@version"
    import bisect
    times = [t for t, _ in arm_at]
    stats = collections.defaultdict(lambda: {"home_s": 0.0, "seen_s": 0.0, "moves": 0})
    obs = d["obs"]
    for a, b in zip(obs, obs[1:]):
        i = bisect.bisect_right(times, a[0]) - 1
        if i < 0 or a[0] - times[i] > 30:
            continue
        arm = arm_at[i][1]
        dt = min(b[0] - a[0], 30)
        if a[2]:
            stats[arm]["home_s"] += dt
            if a[1] >= 0.3:
                stats[arm]["seen_s"] += dt
    for t, act, arm in d["track"]:
        arm = arm.split("@")[0]
        if act and act.split(":")[0] in ("track", "chase", "centre resting pet"):
            stats[arm]["moves"] += 1
        elif act and act.startswith("search"):
            stats[arm]["moves"] += 1
    out = {}
    for arm, s in stats.items():
        h = s["home_s"] / 3600
        out[arm] = {"home_hours": round(h, 1), "logged_in_view": round(s["seen_s"] / max(s["home_s"], 1e-9), 3),
                    "moves_per_home_hour": round(s["moves"] / max(h, 1e-9), 1)}
    print(json.dumps(out, indent=1))
    b = out.get("base")
    files = {"learned": "models/follow_policy.json", "ppo": "models/follow_ppo.onnx"}
    for arm, l in out.items():
        if arm == "base":
            continue
        if not b or min(b["home_hours"], l["home_hours"]) < 8:
            print(f"verdict {arm}: not enough data yet (need 8 home hours per arm)")
            continue
        keep = l["logged_in_view"] >= b["logged_in_view"] and l["moves_per_home_hour"] <= b["moves_per_home_hour"]
        print(f"verdict {arm}:", "KEEP" if keep else f"REMOVE {files.get(arm, arm)} and redeploy")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 3)
