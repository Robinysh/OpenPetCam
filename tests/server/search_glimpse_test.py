"""A search aims at a single strong glimpse and looks again instead of moving on (report 2026-11-04 17:40).

Evidence (search record 2026-11-04/124035, frames looked at by eye): she lay by the bowls at the top edge of the
third stop, cut off by the frame. The verifier scored her 0.977 on one of 4 frames, the other three missed her (the
detector's raw score was 0.157, just over its floor), so the 2-of-4 rule was not met and the search drove on while
she was in view. The rule stays (it keeps a single infrared frame of a slipper from ending a search); what changes is
that a glimpse is followed up: aim at it, then judge fresh frames with her fully in view.
  python3 tests/server/search_glimpse_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
from tracker_test import FH, FW, World, tracker_for  # noqa: E402  (runs the tracker suite once on import)


class EdgeWorld(World):
    """The detector sees her on every frame once she is well inside the picture, but at the frame's edge only on
    one frame in four (cut off, as on 2026-11-04)."""

    def presets(self):
        return [{"x": 400, "y": 500}, {"x": 3200, "y": 500}]

    def find_pet(self, after):
        self.sleep(0.4)
        b = self.box()
        if b is None:
            return None
        f, g = (self.X - self.pet) / FW, (self.Y - self.pet_y) / FH
        if max(abs(f), abs(g)) > 0.33:
            self.edge_calls = getattr(self, "edge_calls", 0) + 1
            return b if self.edge_calls % 4 == 1 else None
        return b


# 1. At a saved view she lies at the top edge: the search aims at the glimpse and finds her there.
w = EdgeWorld(X=1850, pet=400, pet_y=500 + 0.40 * FH)
tr = tracker_for(w)
res = tr.search("test")
assert "found her" in res and "after aiming at a glimpse" in res, res
assert w.box() is not None and abs((w.Y - w.pet_y) / FH) < 0.33, ("not brought into view", w.X, w.Y)
print("edge of a saved view:", res)

# 2. In the view the camera is already in, at its edge: no drive to the saved views; a longer look, then an aim.
w = EdgeWorld(X=1850, pet=1850 + 0.40 * FW)
tr = tracker_for(w)
res = tr.search("test")
assert "found her" in res and "where the camera already was" in res, res
assert not any(m[0] == "goto" for m in w.moves), ("drove away from a glimpse of her", w.moves)
print("edge of the current view:", res, "; moves:", w.moves)

# 3. A weak glimpse (the toy under the stool at night scored 0.86-0.956) is not followed up: no aim at it.
class ToyWorld(EdgeWorld):
    def find_pet(self, after):
        b = EdgeWorld.find_pet(self, after)
        if b is None:
            return None
        v = type("VB", (list,), {})(b)
        v.p = 0.9
        return v


w = ToyWorld(X=1850, pet=400, pet_y=500 + 0.40 * FH)
tr = tracker_for(w)
res = tr.search("test")
assert "after aiming at a glimpse" not in res, res
print("weak glimpse not followed up:", res)

# 4. Fully in view but the detector catches her on one frame in four (lying still, 2026-11-03 20:54 and
#    2026-11-04 09:35: 3 and 8 sightings, never two in one look, "not found"). Sightings now add up across looks.
class FlakyWorld(EdgeWorld):
    def find_pet(self, after):
        self.sleep(0.4)
        self.calls = getattr(self, "calls", 0) + 1
        b = self.box()
        return b if b is not None and self.calls % 4 == 1 else None


w = FlakyWorld(X=1850, pet=400)
tr = tracker_for(w)
res = tr.search("test")
assert "found her" in res and "(400, 500)" in res, res
print("flaky detector, sightings add up:", res)


class FlakyToy(FlakyWorld):
    def find_pet(self, after):
        b = FlakyWorld.find_pet(self, after)
        if b is None:
            return None
        v = type("VB", (list,), {})(b)
        v.p = 0.9
        return v


w = FlakyToy(X=1850, pet=400)
tr = tracker_for(w)
res = tr.search("test")
assert "not found" in res, ("a weak, flaky sighting (the night toy) was confirmed", res)
print("weak flaky sighting not confirmed:", res)

# 5. Nothing anywhere: no glimpse, no extra aims; the search ends as before.
w = EdgeWorld(X=1850, pet=99999)
tr = tracker_for(w)
res = tr.search("test")
assert "not found" in res and not any(m[0] == "aim" for m in w.moves), (res, w.moves)
print("nobody there:", res)
print("search glimpse ok")
