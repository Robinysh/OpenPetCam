"""A running search says whether it is judging the view or moving the camera.

An owner tapped Find and saw "Looking for <pet>" over a motionless picture: the search looks at the current view
before any motor move, and stops at each view to judge it, so the camera is still for much of a search. The tracker
exposes the phase at every tick so the watcher can publish it and the app can say what is happening.
  python3 tests/server/search_phase_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import tracker as T  # noqa: E402
from tracker_test import World  # noqa: E402  (runs the tracker suite once on import)


class SavedViews(World):
    def presets(self):
        return [{"x": 400, "y": 500}]


w = SavedViews(X=1850, pet=-99999)
seen = []
tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now,
               tick=lambda: seen.append((tr.search_phase, w.busy(), len(w.moves))))
assert tr.search_phase is None, "no phase before a search"
tr.search("you asked")

assert seen, "the search never ticked"
assert seen[0] == ("looking", False, 0), ("the first tick must say it is judging this view, before any move", seen[0])
assert any(p == "moving" and busy for p, busy, _ in seen), ("no tick said the camera was moving", seen)
assert any(p == "looking" and n > 0 for p, _, n in seen), ("no tick said it was judging a later view", seen)
assert not any(p == "looking" and busy for p, busy, _ in seen), ("said 'checking this view' while the motor ran", seen)
assert tr.search_phase is None, "the phase must clear when the search ends"

# A Follow aim outside a search is not a search phase.
w = SavedViews(X=1850, pet=-99999)
tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now)
tr._move(w.goto, 400, 500)
assert tr.search_phase is None, "a move outside a search set a search phase"
print("search phase ok")
