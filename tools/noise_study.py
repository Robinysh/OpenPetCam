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


LOUD = r"""
import json, math, os, subprocess, sys, time, wave
sys.path.insert(0, "/opt/petcam")
import app
S = "/var/lib/petcam"
def safe():
    if open(f"{S}/privacy").read().strip() == "1":
        return "paused"
    d = json.load(open(f"{S}/live.json"))
    if d.get("status") != "ok" or not d.get("audio_ok"):
        return "watcher/audio not ok"
    if time.time() - (d.get("last_seen") or 0) < 600:
        return "the dog seen in the last 10 min"
    return None
why = safe()
if why:
    print(json.dumps({"skipped": why})); sys.exit(0)
wav = "/tmp/noise_study.wav"
rec = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-rtsp_transport", "tcp", "-allowed_media_types", "audio",
                        "-i", "rtsp://127.0.0.1:8554/cam", "-ac", "1", "-ar", "16000", wav], stdin=subprocess.DEVNULL)
time.sleep(4)
t0 = time.time()
x0, y0 = app.position()
dx = 300 if x0 + 300 <= 3600 else -300
marks = []
for speed in (100, 150, 250, 400, 900):
    for rep in range(3):
        for tx in (x0 + dx, x0):
            if safe():
                break
            a = time.time(); app.goto(tx, y0, speed); app.wait_idle(); b = time.time()
            marks.append([speed, a - t0, b - t0])
            time.sleep(3)
time.sleep(1)
rec.terminate(); rec.wait(5)
w = wave.open(wav); fr = w.getframerate(); raw = w.readframes(w.getnframes()); w.close(); os.remove(wav)
import array
pcm = array.array("h", raw)
lag = 4.0   # recording started 4 s before t0
def db(a, b):
    i, k = int((a + lag) * fr), int((b + lag) * fr)
    seg = pcm[max(0, i):max(0, k)]
    if len(seg) < fr // 10:
        return None
    blocks = [seg[j:j + fr // 10] for j in range(0, len(seg) - fr // 10 + 1, fr // 10)]
    vals = [20 * math.log10(max(1e-9, math.sqrt(sum(v * v for v in bl) / len(bl)) / 32768)) for bl in blocks]
    return vals
out = []
for speed, a, b in marks:
    move, before = db(a, b), db(a - 2.0, a - 0.2)
    if move and before:
        floor = sorted(before)[len(before) // 2]
        out.append({"speed": speed, "secs": round(b - a, 1), "excess_mean": round(sum(move) / len(move) - floor, 1),
                    "excess_peak": round(max(move) - floor, 1), "floor": round(floor, 1)})
print(json.dumps({"x0": x0, "moves": out}))
"""


def loudness():
    r = subprocess.run(SSH + ["sudo -u petcam /opt/petcam/venv/bin/python -"], input=LOUD, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-400:])
    d = json.loads(r.stdout.strip().splitlines()[-1])
    if "skipped" in d:
        return d
    import statistics
    by = {}
    for m in d["moves"]:
        by.setdefault(m["speed"], []).append(m)
    d["summary"] = {sp: {"n": len(v), "excess_mean_db": round(statistics.median(x["excess_mean"] for x in v), 1),
                         "excess_peak_db": round(statistics.median(x["excess_peak"] for x in v), 1),
                         "secs_per_300_steps": round(statistics.median(x["secs"] for x in v), 1)} for sp, v in sorted(by.items())}
    return d


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "history"
    days = float(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 7
    if cmd == "history":
        print(json.dumps(history(days), indent=1))
    elif cmd == "loudness":
        print(json.dumps(loudness(), indent=1))
