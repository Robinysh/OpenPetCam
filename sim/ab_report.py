#!/usr/bin/env python3
"""Gate 3, live: compare the learned Follow policy with today's constants on the real camera, from the watcher's own
logs. The watcher alternates them in 2-hour blocks, shifted one block per day (server/watch.py apply_policy), and
records which one ran in track.policy and how many motor commands each cycle started in track.moves.

    python3 sim/ab_report.py [days] [--since <code version prefix>]

What it measures (revised 2026-11-04 after an outside review):
  * in view = the share of ALL watched time with a fresh sighting. NOT "while home": "home" was inferred from the same
    detections being judged (absent after 180 s unseen), so an arm that lost her for long stretches dropped those
    stretches from its own denominator and could score higher by finding her less (a 25%-visible arm reported 95%).
    Her absences are not caused by the policy, so over enough balanced blocks they cancel; the comparison is made
    hour-of-day by hour-of-day, on the hours both arms covered, so her daily routine falls on both alike.
  * motor commands per watched hour, every one (a search used to count as one move however many it made).
  * code versions are listed, not silently pooled; --since keeps only cycles from versions starting with the prefix.
Verdict rule: KEEP the learned arm only if, over at least MIN_SHARED_HOURS shared hour-of-day cells and MIN_HOURS
watched hours per arm, its in-view share is not lower (the bootstrap 90% interval of learned - base stays above
-MARGIN) and its motor commands per hour are not higher than base's; otherwise REMOVE. Telemetry, not ground truth:
labelled spot checks remain the real test.
"""
import collections
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_HOURS = 8.0
MIN_SHARED_HOURS = 6
MIN_CELL_S = 20 * 60          # an hour-of-day cell counts for an arm once it has this much watched time
MARGIN = 0.02
SEEN = 0.30

Q = r'''
import sqlite3, json, time, sys
days = float(sys.argv[1]) if len(sys.argv) > 1 else 3
c = sqlite3.connect("file:/var/lib/petcam/petcam.db?mode=ro", uri=True)
t0 = time.time() - days * 86400
cols = [r[1] for r in c.execute("pragma table_info(track)")]
mv = "moves" if "moves" in cols else "null"
print(json.dumps({"obs": c.execute("select ts,score from obs where ts>? order by ts", (t0,)).fetchall(),
                  "track": c.execute(f"select ts,act,policy,{mv} from track where ts>? and policy is not null order by ts",
                                     (t0,)).fetchall(),
                  "tz": time.localtime().tm_gmtoff}))
'''


def score(obs, track, since=None, tz=0):
    """obs: [(ts, score)], track: [(ts, act, 'arm@version', moves|None)]. Returns per-arm stats and the paired
    hour-of-day comparison (pure; tests/sim/ab_report_test.py)."""
    import bisect
    rows = [r for r in track if since is None or r[2].split("@", 1)[-1].startswith(since)]
    times = [r[0] for r in rows]
    arms = collections.defaultdict(lambda: {"watched_s": 0.0, "seen_s": 0.0, "moves": 0, "moves_known_s": 0.0,
                                            "versions": collections.Counter()})
    cells = collections.defaultdict(lambda: [0.0, 0.0])          # (arm, hour of day) -> [watched_s, seen_s]
    for a, b in zip(obs, obs[1:]):
        i = bisect.bisect_right(times, a[0]) - 1
        if i < 0 or a[0] - times[i] > 30:                        # no Follow cycle near: not part of the A/B
            continue
        arm = rows[i][2].split("@")[0]
        dt = min(b[0] - a[0], 30)
        s = arms[arm]
        s["watched_s"] += dt
        hour = int(((a[0] + tz) % 86400) // 3600)
        cells[(arm, hour)][0] += dt
        if a[1] >= SEEN:
            s["seen_s"] += dt
            cells[(arm, hour)][1] += dt
    for t, act, pol, moves in rows:
        arm, _, ver = pol.partition("@")
        s = arms[arm]
        s["versions"][ver] += 1
        if moves is not None:
            s["moves"] += moves
            s["moves_known_s"] += 1                              # cycles with a moves count
    out = {}
    for arm, s in arms.items():
        h = s["watched_s"] / 3600
        out[arm] = {"watched_hours": round(h, 1), "in_view": round(s["seen_s"] / max(s["watched_s"], 1e-9), 3),
                    "motor_commands": s["moves"], "versions": dict(s["versions"].most_common(5))}
    # moves per hour only over the cycles that logged them (the column is new)
    for arm, s in arms.items():
        known = [r for r in rows if r[2].split("@")[0] == arm and r[3] is not None]
        if len(known) > 1:
            span = sum(min(b[0] - a[0], 30) for a, b in zip(known, known[1:])) / 3600
            out[arm]["motor_commands_per_hour"] = round(s["moves"] / max(span, 1e-9), 1)
    return out, cells


def paired(cells, a, b, n_boot=2000, seed=0):
    """Mean in-view difference a - b over hour-of-day cells both arms covered, with a bootstrap 90% interval."""
    hours = sorted(h for (arm, h) in cells if arm == a and cells[(a, h)][0] >= MIN_CELL_S
                   and cells.get((b, h), [0, 0])[0] >= MIN_CELL_S)
    if not hours:
        return None
    d = [cells[(a, h)][1] / cells[(a, h)][0] - cells[(b, h)][1] / cells[(b, h)][0] for h in hours]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {"shared_hours": len(hours), "diff": round(sum(d) / len(d), 3),
            "ci90": [round(boots[int(0.05 * n_boot)], 3), round(boots[int(0.95 * n_boot)], 3)]}


def verdict(out, cells, arm, base="base"):
    b, l = out.get(base), out.get(arm)
    if not b or not l or min(b["watched_hours"], l["watched_hours"]) < MIN_HOURS:
        return "not enough data yet", None
    p = paired(cells, arm, base)
    if not p or p["shared_hours"] < MIN_SHARED_HOURS:
        return f"not enough data yet (need {MIN_SHARED_HOURS} hours of the day covered by both arms)", p
    mpl, mpb = l.get("motor_commands_per_hour"), b.get("motor_commands_per_hour")
    quieter = mpl is None or mpb is None or mpl <= mpb
    keep = p["ci90"][0] >= -MARGIN and p["diff"] >= 0 and quieter
    return ("KEEP" if keep else "REMOVE"), p


def main(days=3, since=None):
    sys.path.insert(0, os.path.join(HERE, "..", "tools"))
    from siteconf import SSH
    raw = subprocess.run(SSH + [f"sudo python3 - {days}"], input=Q, capture_output=True, text=True, timeout=300, check=True).stdout
    d = json.loads(raw)
    out, cells = score(d["obs"], d["track"], since, d.get("tz", 0))
    print(json.dumps(out, indent=1))
    files = {"learned": "models/follow_policy.json", "ppo": "models/follow_ppo.onnx"}
    for arm in out:
        if arm == "base":
            continue
        v, p = verdict(out, cells, arm)
        extra = f" (in view {p['diff']:+.3f}, 90% {p['ci90']}, {p['shared_hours']} shared hours)" if p else ""
        print(f"verdict {arm}: {v}{extra}" + (f" -> remove {files.get(arm, arm)} and redeploy" if v == "REMOVE" else ""))


if __name__ == "__main__":
    a = sys.argv[1:]
    since = a[a.index("--since") + 1] if "--since" in a else None
    nums = [x for x in a if x.replace(".", "", 1).isdigit()]
    main(float(nums[0]) if nums else 3, since)
