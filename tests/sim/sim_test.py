"""The Follow simulator is a measuring instrument: these checks hold it to the measured hardware and to its own physics.

    python3 tests/sim/sim_test.py
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim"))
import gate1  # noqa: E402
from room import Room, load_params  # noqa: E402

P = load_params()


def trace(seed):
    r = Room(seed=seed, free=P.get("free"))
    T = gate1.load_tracker()
    obs, acts = gate1.run_watcher(r, T, 0.5)
    return hashlib.sha256(json.dumps([obs, acts, r.moves], default=str).encode()).hexdigest()


# 1. Determinism: one seed, one history
assert trace(3) == trace(3), "same seed gave different histories"
assert trace(3) != trace(4), "different seeds gave the same history"
print("determinism: same seed -> identical run, different seed -> different run")

# 2. Geometry: in view exactly when her centre is within 0.45 frame of the image centre; aiming centres her
r = Room(seed=0)
r.dog_state, r.px, r.py = "rest", r.X + 0.3 * r.FW, r.Y
assert r.in_view()
r.px = r.X + 0.46 * r.FW
assert not r.in_view()
r.px, r.py = r.X - 0.3 * r.FW, r.Y
f, g = r.offset()
r.aim(f, g)
r.wait_idle()
f2, g2 = r.offset()
assert abs(f2) < 0.5 * r.backlash / r.FW + 0.005 and abs(g2) < 0.01, (f2, g2)   # within half the gear play
print("geometry: visibility boundary at 0.45 frame; aim(offset) centres her to within half the gear play")

# 3. Motor: measured quiet speed (218 steps/s), ~0.1 s start delay, a new move waits for the one in progress
r = Room(seed=0)
x0, t0 = r.X, r.t
r.goto(x0 + 400, r.Y)
r.wait_idle()
elapsed = r.t - t0
assert abs(elapsed - (0.1 + 400 / 218)) < 0.12, elapsed
t1 = r.t
r.goto(r.X + 600, r.Y)
r.goto(r.X - 100, r.Y)            # issued mid-move: starts only after the first move finished (measured: queues)
assert r.t - t1 >= 600 / 218, ("second command did not wait", r.t - t1)
print(f"motor: 400 steps in {elapsed:.2f} s (measured 1.76-1.83), a new command waits for the move in progress")

# 3b. Gear play (measured 27 steps): after a reversal the counter moves at once, the picture only after the play
r = Room(seed=0)
r.goto(r.C + 200, r.Y); r.wait_idle()
c0, x0 = r.C, r.X
r.goto(r.C - 200, r.Y); r.wait_idle()
assert abs((c0 - r.C) - 200) < 1 and abs((x0 - r.X) - (200 - r.backlash)) < 1, (c0 - r.C, x0 - r.X)
print(f"gear play: a 200-step reversal moves the counter 200 and the picture {x0 - r.X:.0f} (measured 150/173 px ratio)")
# 4. Sensing: detection rates match the fitted ones within sampling error; frames are 0.7 s late
r = Room(seed=1)
r.dog_state, r.px, r.py = "rest", r.X, r.Y
r.state_until = 1e18
n = 4000
hits = 0
for _ in range(n):
    r.advance(0.05)
    hits += r.frame()[0] is not None
p = P["detect_persist"]["still"]["p"]
se = (p * (1 - p) / n) ** 0.5
assert abs(hits / n - p) < 4 * se, (hits / n, p)
r = Room(seed=2)
r.dog_state, r.px, r.py, r.state_until = "rest", r.X, r.Y, 1e18
r.advance(1.0)
x_start = r.X
r.goto(r.X + 733, r.Y)            # turn one frame away
r.advance(0.5)                    # the motor has moved, but a frame arriving now was taken 0.7 s ago
assert r.X > x_start + 50 and r._past(r.cam_hist, r.F["lag_s"])[1] == x_start, (r.X, r._past(r.cam_hist, 0.7))
r.advance(1.0)
assert r._past(r.cam_hist, r.F["lag_s"])[1] > x_start + 50
# 4b. Joint exposure time: a frame shows the camera AND the pet as they were lag_s ago, never a mix
r = Room(seed=5, free={"p_false_accept": 0.0})       # no lookalikes: only she can produce a box here
r.dog_state, r.state_until = "away", 1e18
r.advance(2.0)
r.dog_state, r.px, r.py = "rest", r.X, r.Y            # she appears dead centre right now
assert all(r.frame()[0] is None for _ in range(30)), "a frame showed her before her appearance had reached the camera"
r.advance(r.F["lag_s"] + 0.2)
assert any(r.frame()[0] is not None for _ in range(30)), "she never appeared once the frames caught up"
print(f"sensing: still-dog detection {hits / n:.3f} vs fitted {p:.3f}; frames lag {r.F['lag_s']} s")

# 5. Gate 1 on two held-out seeds (short): the real controller's logged stats stay in a plausible band
rows = []
T = gate1.load_tracker()
for s in (200, 201):
    rr = Room(seed=s, free=P.get("free"))
    obs, acts = gate1.run_watcher(rr, T, 2)
    rows.append(gate1.metrics(obs, acts))
iv = sum(m["in_view_share_while_home"] for m in rows) / len(rows)
assert 0.4 < iv < 0.98, iv
print(f"gate 1 smoke: real controller in the sim, logged in-view {iv:.2f} (real install 0.756)")
print("sim ok")
