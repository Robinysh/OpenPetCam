"""A mid-pan sighting is placed where the camera pointed when its frame was taken, and the search goes back to her
(report 2026-11-04 14:45: she lay in plain view; three sure sightings while panning, but each frame arrived ~1.5-2 s
after it was taken, the camera had panned on ~0.5 frame, every look at the stop showed the next view, "not found").
Here the frame delay is 2 s and the detector only sees her while she is well inside the picture.
  python3 tests/server/search_overshoot_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
from tracker_test import FW, World, tracker_for  # noqa: E402  (runs the tracker suite once on import)


class Box(list):
    pass


class LateWorld(World):
    LAG = 2.0

    def box_at(self, X, Y):
        b = World.box_at(self, X, Y)
        return b if b is not None and abs((X - self.pet) / FW) < 0.35 else None   # not at the very edge

    def find_now(self, since):
        past = [h for h in self.hist if h[0] <= self.t - self.LAG]
        t, X, Y = past[-1] if past else self.hist[0]
        b = self.box_at(X, Y)
        if b is None:
            return None
        v = Box(b)
        v.p, v.captured = 1.0, t
        return v

    def find_pet(self, after):
        self.sleep(0.4)
        return self.box()


w = LateWorld(X=200, pet=1300)
tr = tracker_for(w)
res = tr.search("test")
assert "found her" in res and "panning" in res, (res, w.moves)
import re  # noqa: E402
secs = int(re.search(r"in (\d+) s", res).group(1))
assert secs <= 15, ("the first sighting must be followed up at once, not after several more stops", res)
assert abs((w.X - w.pet) / FW) < 0.3, ("not brought back onto her", w.X, w.pet)
print("late frames while panning:", res, "| camera at", round(w.X), "she is at", w.pet)
print("search overshoot ok")
