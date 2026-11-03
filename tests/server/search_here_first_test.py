"""Search looks where the camera already is before it moves (goal search_success, 2026-11-02).

Evidence (2026-11-02, frames labelled by eye): at 13:45 and 18:31 the dog lay asleep in the camera's view, the
detector missed her for 48 s / 79 s (a dropout on a still dog; the second right after a watcher restart),
and the timer search started by driving to the saved views. It came back "not found" and she was
detected in the same view 38 s / 9 s later. Looking at the current view first finds her with no motor
move at all, which is also what the owners want near a resting dog.
  python3 tests/server/search_here_first_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
from tracker_test import World, tracker_for  # noqa: E402  (runs the tracker suite once on import)


class SavedViews(World):
    def presets(self):
        return [{"x": 400, "y": 500}, {"x": 3200, "y": 500}]


# 1. Asleep in view, slightly off-centre; the detector misses her on every cycle for 60 s (dropout), then a
#    fresh look sees her. The search must find her where the camera is, and not move the camera at all.
w = SavedViews(X=1850, pet=1850 - 0.25 * 733)
tr = tracker_for(w)
log = []
for _ in range(60):
    w.advance(1.0)
    a = tr.step(None, moving=False)              # the watcher's detector misses her (dropout)
    if a:
        log.append(a)
s = [a for a in log if a.startswith("search")]
assert s and "found her where the camera already was" in s[0], ("moved away before looking here", s, w.moves)
assert w.moves == [], ("motor moved near a resting dog that was already in view", w.moves)
print("asleep in view, detector dropout:", s[0], "; moves:", w.moves)

# 2. Not in the current view: the saved views and the sweep still run as before and find her.
w = SavedViews(X=1850, pet=3200)
tr = tracker_for(w)
res = tr.search("test")
assert "found her at the saved view (3200, 500)" in res, res
print("elsewhere:", res)

# 3. Owner's "Find the dog" while she is already in view but the cycle's detection missed: found, no move.
w = SavedViews(X=1000, pet=1000)
tr = tracker_for(w)
res = tr.search("you asked")
assert "where the camera already was" in res and w.moves == [], (res, w.moves)
print("asked, she is in view:", res)
print("search here first ok")
