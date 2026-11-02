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
print("presence ok:", events)
