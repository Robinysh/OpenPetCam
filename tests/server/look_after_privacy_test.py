"""After privacy ends, one still look (several frames, no motor move) runs as soon as the live stream is back, so a
resting pet the single-frame detector misses is not reported as "isn't in view" (report 20261107-100906).
  python3 tests/server/look_after_privacy_test.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import watch  # noqa: E402
from tracker_test import World, tracker_for  # noqa: E402

clock = [1000.0]
watch.now = lambda: clock[0]
watch.log = lambda *a: None
w = watch.Watch.__new__(watch.Watch)
world = World(X=1850, pet=1850)
w.tracker = tracker_for(world)
w.privacy_on = lambda: False
w.present, w.source = False, "stream"
seen = []
w.update_pet = lambda con, t, score, moving, jpg, box=None: seen.append((score, box)) or setattr(w, "present", True)
con = types.SimpleNamespace()

# not due yet, then due with the stream: one look, she is there, no motor move
w.look_here_at = 1003.0
w.look_after_privacy(con, 1001.0, b"", 0.0)
assert not seen and w.look_here_at == 1003.0
w.look_after_privacy(con, 1004.0, b"", 0.0)
assert seen and seen[0][0] >= watch.PRESENT_SCORE and w.look_here_at is None, seen
assert world.moves == [], world.moves
print("still look after privacy: found her without moving")

# runs once only; skipped when the detector already sees her or the stream is not back within 2 min
seen.clear()
w.look_after_privacy(con, 1010.0, b"", 0.0)
assert not seen
w.present, w.look_here_at = False, 2000.0
w.source = "snapshot"
w.look_after_privacy(con, 2005.0, b"", 0.0)
assert w.look_here_at == 2000.0 and not seen          # waits for live frames
w.look_after_privacy(con, 2200.0, b"", 0.0)
assert w.look_here_at is None and not seen            # gave up after 2 min
print("look after privacy ok")
