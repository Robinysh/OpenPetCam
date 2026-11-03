"""Presence state with simulated detections: 'resting' must not outlive her leaving the frame.

Regression for 2026-11-02 09:40: she walked out at 09:39:30 and the app said "resting 5m" over an
empty hallway until the 3-minute leave debounce ended. Runs on the server or locally:
  python3 tests/server/presence_test.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, "/opt/petcam")
import watch  # noqa: E402

w = watch.Watch.__new__(watch.Watch)
w.present, w.present_since, w.rest_since, w.moving_since = False, None, None, None
w.last_seen, w.gone_since = 0.0, None
import collections
w.cam_moves, w.move_box0 = collections.deque(maxlen=20), None
watch.MOVED = "/nonexistent/moved_at"   # no camera moves in this replay
events = []
watch.add_event = lambda con, kind, detail, thumb, ts: events.append((kind, ts)) or len(events)
w.save_thumb = lambda *a: None
w.clips = types.SimpleNamespace(want=lambda *a: None)

t = 1000.0
def step(score, moving, n):
    global t
    for _ in range(n):
        w.update_pet(None, t, score, moving, b"")
        t += 3

step(1.0, False, 100)            # 5 min resting in view
assert w.present and w.rest_since and not w.moving_since, "should be resting"
step(1.0, False, 1); step(0.0, False, 1); step(1.0, False, 1)   # one missed detection
assert w.rest_since == 1000.0, f"a single miss must not reset rest ({w.rest_since})"
step(1.0, True, 3)               # gets up
step(0.0, True, 2)               # walks out (motion, no detection)
step(0.0, False, 12)             # empty hallway for 36 s
assert w.present, "still within the leave debounce"
assert w.rest_since is None and w.moving_since is None, f"out of sight must not be resting/moving: {w.rest_since}, {w.moving_since}"
step(0.0, False, 60)             # 3 more minutes
assert not w.present and events[-1][0] == "left", events

# Report 2026-11-02 19:26:44: she appeared at 19:24:57 and spent the next 90 s moving around right in
# front of the camera; one still frame at 19:26:21 ended the movement run and the app said
# "resting 2m", counting from her arrival. Real obs (seconds after arrival, moving, box) replayed:
# after a real movement run (6 s+, box shifted), rest counts from the next still frame.
w.present, w.gone_since, w.last_seen, w.rest_since, w.moving_since = False, 0.0, 0.0, None, None
T0 = 50000.0
obs = [(0, 1, (0.00, 0.00, 0.24, 0.98)), (37, 0, (0.00, 0.00, 0.56, 1.00)),
       (44, 1, (0.25, 0.00, 0.87, 0.99)), (47, 1, (0.23, 0.00, 0.85, 0.99)),
       (49, 1, (0.30, 0.00, 0.87, 0.98)), (52, 1, (0.36, 0.00, 0.87, 0.99)),
       (57, 1, (0.12, 0.00, 0.76, 1.00)), (63, 1, (0.13, 0.06, 0.79, 0.98)),
       (72, 1, (0.32, 0.00, 0.85, 0.98)), (78, 1, (0.36, 0.00, 0.84, 0.98)),
       (84, 0, (0.33, 0.00, 0.88, 0.98))]
for dt, mv, box in obs:
    w.update_pet(None, T0 + dt, 1.0, bool(mv), b"", box)
assert w.present and not w.moving_since
assert w.rest_since == T0 + 84, f"rest must count from the still frame after she moved, not her arrival: {w.rest_since - T0:+.0f} s"
# ...but a short twitch during a long rest (no 6 s movement run) keeps the rest going.
w.update_pet(None, T0 + 400, 1.0, False, b"", (0.33, 0.00, 0.88, 0.98))
w.update_pet(None, T0 + 403, 1.0, True, b"", (0.33, 0.00, 0.88, 0.98))
w.update_pet(None, T0 + 406, 1.0, False, b"", (0.33, 0.00, 0.88, 0.98))
assert w.rest_since == T0 + 84, "a twitch must not restart the rest"
print("presence ok:", events)
