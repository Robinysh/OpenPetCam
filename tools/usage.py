#!/usr/bin/env python3
"""How the app is really used, from the camera server's usage table (written by app.py).

  usage.py [--days 7] [--json]

Prints sessions, what gets used, joystick gesture shape (and how often a move is corrected back,
i.e. overshoot), response times per action, how long live video takes to appear, stalls and
reconnects, failures, and the friction signals worth fixing. The nightly check-in reads the JSON.
Nothing here leaves your machines: the data lives on your server, this reads it over SSH.
"""
import collections
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from siteconf import SSH  # noqa: E402


def fetch(days):
    q = ("import sqlite3,json,time; c=sqlite3.connect('/var/lib/petcam/petcam.db'); "
         f"print(json.dumps(c.execute('select ts,device,kind,name,ms,ok,detail from usage where ts>?',(time.time()-{days}*86400,)).fetchall()))")
    out = subprocess.run(SSH + [f'sudo python3 -c "{q}"'], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"could not read usage: {out.stderr[-300:]}")
    return json.loads(out.stdout or "[]")


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))]) if xs else None


def report(days=7):
    rows = fetch(days)
    api = [r for r in rows if r[2] == "api"]
    ui = [r for r in rows if r[2] == "ui"]
    det = lambda r: json.loads(r[6]) if r[6] else {}   # noqa: E731

    # sessions: page-side session_end carries its length; count starts for glance vs long use
    ends = [det(r).get("secs", 0) for r in ui if r[3] == "session_end"]
    starts = sum(1 for r in ui if r[3] == "session_start")
    sessions = {"starts": starts, "ended": len(ends), "median_secs": pct(ends, .5), "p90_secs": pct(ends, .9),
                "glances_under_30s": sum(1 for x in ends if x < 30)}

    # features used: API actions + page events
    used = collections.Counter(r[3] for r in api) + collections.Counter(
        r[3] + (":" + det(r).get("tab") if r[3] == "tab" else "") + (":" + det(r).get("name") if r[3] == "sheet" else "")
        for r in ui if r[3] not in ("session_start", "session_end", "joystick", "live_first_frame"))

    # joystick gestures
    js = [r for r in ui if r[3] == "joystick"]
    jd = [det(r) for r in js]
    joystick = {"gestures": len(js), "median_ms": pct([r[4] for r in js], .5), "median_ticks": pct([d.get("ticks") for d in jd], .5),
                "directions": dict(collections.Counter(d.get("dir") for d in jd)),
                "overshoot_rate": round(sum(1 for d in jd if d.get("reversal")) / len(jd), 2) if jd else None,
                "full_tilt_rate": round(sum(1 for d in jd if (d.get("peak") or 0) > 0.9) / len(jd), 2) if jd else None}

    # response times per action (server side) and failures
    by = collections.defaultdict(list)
    for r in api:
        by[r[3]].append(r[4])
    latency = {k: {"n": len(v), "p50_ms": pct(v, .5), "p90_ms": pct(v, .9)} for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}
    failures = collections.Counter(f"{r[3]} -> {det(r).get('status')}" for r in api if not r[5])

    first = [r[4] for r in ui if r[3] == "live_first_frame"]
    video = {"starts": len(first), "first_frame_p50_ms": pct(first, .5), "first_frame_p90_ms": pct(first, .9),
             "stalls": sum(1 for r in ui if r[3] == "live_stall"), "reconnects": sum(1 for r in ui if r[3] == "live_reconnect")}

    friction = []
    if joystick["overshoot_rate"] and joystick["overshoot_rate"] > 0.25:
        friction.append(f"{joystick['overshoot_rate']:.0%} of joystick moves are corrected back right away: moves overshoot; consider smaller steps near the centre")
    if video["first_frame_p90_ms"] and video["first_frame_p90_ms"] > 3000:
        friction.append(f"live video takes {video['first_frame_p90_ms'] / 1000:.1f} s to appear (p90)")
    if starts and (video["stalls"] + video["reconnects"]) / max(1, starts) > 0.3:
        friction.append(f"video stalls or reconnects in many sessions ({video['stalls']} stalls, {video['reconnects']} reconnects)")
    for k, v in latency.items():
        if v["n"] >= 5 and v["p90_ms"] and v["p90_ms"] > 1500:
            friction.append(f"{k} is slow: p90 {v['p90_ms']} ms")
    if failures:
        friction.append("failed actions: " + ", ".join(f"{k} ×{n}" for k, n in failures.most_common(5)))

    return {"days": days, "events": len(rows), "sessions": sessions, "used": dict(used.most_common(25)), "joystick": joystick,
            "video": video, "latency": dict(list(latency.items())[:12]), "failures": dict(failures), "friction": friction}


def main():
    days = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 7
    r = report(days)
    if "--json" in sys.argv:
        print(json.dumps(r, indent=1))
        return
    s, j, v = r["sessions"], r["joystick"], r["video"]
    print(f"# Usage, last {days} days ({r['events']} events)\n")
    print(f"Sessions: {s['starts']} opened, median {s['median_secs']} s, p90 {s['p90_secs']} s, {s['glances_under_30s']} glances under 30 s")
    print(f"Joystick: {j['gestures']} gestures, median {j['median_ms']} ms / {j['median_ticks']} ticks, overshoot {j['overshoot_rate']}, "
          f"full tilt {j['full_tilt_rate']}, directions {j['directions']}")
    print(f"Live video: {v['starts']} starts, first frame p50 {v['first_frame_p50_ms']} ms / p90 {v['first_frame_p90_ms']} ms, "
          f"{v['stalls']} stalls, {v['reconnects']} reconnects")
    print("\nMost used:")
    for k, n in r["used"].items():
        print(f"  {n:5d}  {k}")
    print("\nResponse times:")
    for k, x in r["latency"].items():
        print(f"  {k:28s} n={x['n']:<5d} p50 {x['p50_ms']} ms  p90 {x['p90_ms']} ms")
    print("\nFriction:" if r["friction"] else "\nFriction: none detected")
    for f in r["friction"]:
        print("  -", f)


if __name__ == "__main__":
    main()
