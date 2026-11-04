"""The owner can stop a running search (report 2026-11-03 07:31: "Looking for <pet> needs to be cancelable").

Evidence: the owner tapped Find at 07:29:32; the sweep ran 127 s over 3 rows and nothing in the app could stop
it (the button was disabled while it ran). A 'Stop looking' press after the search started must end it at
once, stop the motor where it is, and not drive anywhere else (no return move).
  python3 tests/server/search_cancel_test.py
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


def tracker_with_cancel(w, cancel_at):
    """cancel_at: simulated time the owner presses 'Stop looking' (None = never)."""
    return T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now,
                     cancelled=lambda since: cancel_at is not None and since < cancel_at <= w.t)


# 1. Pressed mid-sweep (she is nowhere in reach): the search ends within a second of the press, the motor stops
#    where it is, and there is no move back to the start.
w = SavedViews(X=1850, pet=-99999)
tr = tracker_with_cancel(w, cancel_at=20.0)
res = tr.search("you asked")
assert "stopped, you cancelled it" in res, res
assert w.t < 21.5, ("search kept running after the press", w.t)
assert not w.busy(), "motor still moving after cancel"
assert w.moves[-1][1] != 1850, ("drove back to the start after a cancel", w.moves)
print(f"cancelled mid-sweep: {res} at t={w.t:.1f}; moves {w.moves}")

# 2. Cancelled while driving to a saved view: stops there too.
w = SavedViews(X=3000, pet=-99999)
tr = tracker_with_cancel(w, cancel_at=2.0)
res = tr.search("you asked")
assert "you cancelled it" in res and w.t < 4.0 and not w.busy(), (res, w.t)
print(f"cancelled on the way to a saved view: {res} at t={w.t:.1f}")

# 3. A press from before this search started does not cancel it.
w = SavedViews(X=1850, pet=400)
w.t = 100.0
tr = tracker_with_cancel(w, cancel_at=50.0)
res = tr.search("you asked")
assert "found her at the saved view" in res, res
print("old press ignored:", res)
assert tr.search_t0 is None and not tr.searching
print("search cancel ok")

# 4. Cancelled during the confirming look at a stop where she is visible but off-centre (review 2026-11-04: cancel at
#    8.5 s, then a centring aim at 8.9 s and "found"). No motor command may follow the press.
w = SavedViews(X=1850, pet=400 + 0.3 * 733)
state = {"at": None}


def cancel_on_arrival(since):
    if state["at"] is None and not w.busy() and abs(w.X - 400) < 1:
        state["at"] = w.t + 1.0           # the owner presses Stop during the look (after the settle pause)
    return state["at"] is not None and since < state["at"] <= w.t


tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now,
               cancelled=cancel_on_arrival)
n_moves_at = {}
orig_aim, orig_goto = w.aim, w.goto
w.aim = lambda fx, fy: (n_moves_at.setdefault("aim_t", w.t), orig_aim(fx, fy))
res = tr.search("you asked")
assert "stopped, you cancelled it" in res, res
assert "aim_t" not in n_moves_at or n_moves_at["aim_t"] < state["at"], ("aimed after the cancel", n_moves_at, state)
print("cancelled during the confirming look:", res)
print("search cancel ok (with look boundary)")
