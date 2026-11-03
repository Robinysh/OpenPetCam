"""Follow with realistic detection: frames lag 0.7 s, a moving dog is often scored just under the verifier's
cut-off (motion blur), and a lookalike elsewhere in the room scores weakly too. Measures how much of the time
she stays in view, how many motor moves that costs, and how long the camera sits on the lookalike instead of her.
  python3 tests/server/follow_test.py            new tracker, with assertions
  python3 tests/server/follow_test.py --compare  also the tracker of an older commit (git show <rev>:server/tracker.py)
"""
import importlib.util
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, HERE)
import tracker as T  # noqa: E402
from tracker_test import World, FW  # noqa: E402  the simulated room (no hardware)

LAG, CYCLE = 0.7, 1.0


def detections(w, rnd, decoy, p_moving, p_rest):
    """What one watcher cycle would hand the tracker: a verified box (or None) and weak boxes, from a frame LAG old."""
    past = [h for h in w.hist if h[0] <= w.t - LAG]
    _, X, Y = past[-1] if past else w.hist[0]
    b = w.box_at(X, Y)
    jitter = lambda bb: [v + rnd.uniform(-0.015, 0.015) for v in bb]
    verified, weak = None, []
    if b is not None:
        if rnd.random() < (p_moving if w.moving else p_rest):
            verified = jitter(b)
        elif rnd.random() < 0.8:
            weak.append(jitter(b))
    if decoy is not None:
        f = (X - decoy) / FW
        if abs(f) < 0.45 and rnd.random() < 0.5:
            weak.append([0.5 + f - 0.07, 0.55, 0.5 + f + 0.07, 0.75])
    return verified, weak


def scenario(TR, path, seed, decoy=None, p_moving=0.45, p_rest=0.95, weak_ok=True):
    """path: list of (seconds, speed in steps/s). Returns (share of time in view, moves, seconds centred on the decoy)."""
    rnd = random.Random(seed)
    w = World(X=path[0][2] if len(path[0]) > 2 else 2600, pet=path[0][2] if len(path[0]) > 2 else 2600)
    tr = TR.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now)
    tr.last_seen_t = w.t
    seen = total = on_decoy = 0
    for _ in range(3):                              # resting in view, confirmed, before she sets off
        w.advance(CYCLE); tr.step(w.box(), moving=False)
    for secs, speed, *_ in path:
        for _ in range(int(secs / CYCLE)):
            f_before = w.fx()
            w.advance(CYCLE)
            w.pet += speed * CYCLE
            w.moving = speed != 0
            v, weak = detections(w, rnd, decoy, p_moving, p_rest)
            motion = None                           # the frame reader sees where the picture changed, as in tracker_test
            if speed and abs(f_before) < 0.5:
                motion = (w.t - 0.2, max(-0.49, min(0.49, w.fx())), 0.0)
            try:
                tr.step(v, moving=w.moving and v is not None, weak=weak if weak_ok else None, motion=motion)
            except TypeError:                       # an older tracker without weak boxes
                tr.step(v, moving=w.moving and v is not None, motion=motion)
            total += 1
            seen += w.box() is not None
            if decoy is not None and w.box() is None and abs((w.X - decoy) / FW) < 0.2:
                on_decoy += 1
    return seen / total, len(w.moves), on_decoy


SCENARIOS = {
    "walks across the room": [(25, -110, 3400)],
    "trots, then lies down": [(6, -370, 2900), (60, 0)],
    "walks, stops, walks back": [(8, -150, 2800), (10, 0), (10, 150)],
    "walks past a lookalike": [(12, -140, 3000), (12, 0)],
}
DECOY = {"walks past a lookalike": 1900}


def run_all(TR, seeds=range(10)):
    out = {}
    for name, path in SCENARIOS.items():
        r = [scenario(TR, path, s, decoy=DECOY.get(name)) for s in seeds]
        out[name] = (sum(x[0] for x in r) / len(r), sum(x[1] for x in r) / len(r), sum(x[2] for x in r) / len(r))
    return out


def load_old(rev):
    src = subprocess.run(["git", "show", f"{rev}:server/tracker.py"], capture_output=True, text=True, check=True,
                         cwd=os.path.join(HERE, "..", "..")).stdout
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False); f.write(src); f.close()
    spec = importlib.util.spec_from_file_location("tracker_old", f.name); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


new = run_all(T)
if "--compare" in sys.argv:
    rev = sys.argv[sys.argv.index("--compare") + 1] if len(sys.argv) > sys.argv.index("--compare") + 1 else "HEAD"
    old = run_all(load_old(rev))
    print(f"{'scenario':<28} {'in view: old -> new':>22} {'moves: old -> new':>20} {'s on lookalike':>16}")
    for k in SCENARIOS:
        print(f"{k:<28} {old[k][0]:>10.0%} -> {new[k][0]:<8.0%} {old[k][1]:>9.1f} -> {new[k][1]:<7.1f} {old[k][2]:>6.1f} -> {new[k][2]:.1f}")
for k, (seen, moves, dec) in new.items():
    print(f"follow, {k}: in view {seen:.0%}, {moves:.1f} moves, {dec:.1f} s on the lookalike")
# measured 2026-11-03 against the tracker before weak boxes (git 54d2f80): walk 33% -> 77%, walk-stop-walk 49% -> 62%,
# past a lookalike 41% -> 73%, with no more time on the lookalike (1.4 -> 1.6 s)
assert new["walks across the room"][0] >= 0.65, new
assert new["walks past a lookalike"][0] >= 0.6, new
assert new["walks past a lookalike"][2] <= 3.0, ("drifted onto the lookalike", new)
print("follow ok")
