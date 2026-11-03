#!/usr/bin/env python3
"""Does the camera's own motor disturb the dog? Measured, not guessed (owner decision 2026-11-02: "measure, then decide").

  noise_study.py history [--days 7]    from logs only, nothing moves: her reaction to automatic moves vs a baseline
  noise_study.py loudness              moves the camera (only while she is NOT in view): dB over the room per speed

history: every automatic move (follow: track / chase / centre / search) logged by the watcher, joined with the
presence log. A move "while resting" is one where she was in view and not moving for the 30 s before it. Her
reaction = she is seen moving within 15 s after the move. Baseline = the same 15 s reaction rate at random moments
of rest with no move in the 60 s around them. Ground truth is the watcher's own movement signal, not a model
judging the motor; a rate needs both numbers and their counts to mean anything.
"""
import json
import random
import subprocess
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from siteconf import SSH  # noqa: E402

PULL = r'''
import json, sqlite3, subprocess, time, sys
days = float(sys.argv[1])
since = time.time() - days * 86400
j = subprocess.run(["journalctl", "-u", "petcam-watch", "--since", "@%d" % since, "--no-pager", "-o", "short-unix"],
                   capture_output=True, text=True).stdout
moves = []
for line in j.splitlines():
    if "follow: " not in line:
        continue
    kind = line.split("follow: ", 1)[1].split(":")[0].split(" ")[0]
    if kind in ("track", "chase", "centre", "search"):
        moves.append([float(line.split()[0]), kind])
c = sqlite3.connect("/var/lib/petcam/petcam.db")
obs = c.execute("select ts, present, moving, score from obs where ts > ? order by ts", (since,)).fetchall()
print(json.dumps({"moves": moves, "obs": obs}))
'''


def pull(days):
    r = subprocess.run(SSH + [f"sudo python3 - {days}"], input=PULL, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-300:])
    return json.loads(r.stdout)


def history(days):
    import bisect
    d = pull(days)
    obs = d["obs"]
    ts = [o[0] for o in obs]

    def window(a, b):
        i, k = bisect.bisect_left(ts, a), bisect.bisect_right(ts, b)
        return obs[i:k]

    def resting_at(t):
        w = window(t - 30, t)
        # present and still the whole time, seen at least once (detection blinks while she lies still)
        return len(w) >= 6 and all(o[1] == 1 and not o[2] for o in w) and any(o[3] and o[3] > 0 for o in w)

    def reacted(t):
        return any(o[2] for o in window(t + 1, t + 15))

    moves = [(t, k) for t, k in d["moves"]]
    at_rest = [(t, k) for t, k in moves if resting_at(t)]
    by_kind = {}
    for t, k in at_rest:
        by_kind.setdefault(k, [0, 0])
        by_kind[k][0] += 1
        by_kind[k][1] += reacted(t)
    move_ts = [t for t, _ in moves]
    rng = random.Random(7)
    cand = [o[0] for o in obs if o[1] == 1 and o[3] and o[3] > 0 and not o[2]]
    base = []
    for t in rng.sample(cand, min(4000, len(cand))):
        i = bisect.bisect_left(move_ts, t - 60)
        if i < len(move_ts) and move_ts[i] <= t + 60:
            continue
        if resting_at(t):
            base.append(reacted(t))
    n_rest, r_rest = len(at_rest), sum(v[1] for v in by_kind.values())
    out = {"days": days, "moves_total": len(moves), "moves_while_resting": n_rest,
           "reaction_after_move": round(r_rest / n_rest, 3) if n_rest else None, "by_kind": by_kind,
           "baseline_samples": len(base), "reaction_baseline": round(sum(base) / len(base), 3) if base else None}
    # Wilson 95% intervals: with tens of events a difference can be noise; say so
    import math

    def wilson(k, n):
        if not n:
            return None
        p, z = k / n, 1.96
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [round(c - h, 3), round(c + h, 3)]
    out["ci_after_move"] = wilson(r_rest, n_rest)
    out["ci_baseline"] = wilson(sum(base), len(base))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "history"
    days = float(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 7
    if cmd == "history":
        print(json.dumps(history(days), indent=1))
    else:
        print("loudness: not run yet (needs her out of view; see docs/TODO.md)")
