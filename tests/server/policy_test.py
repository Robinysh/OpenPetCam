"""A learned Follow policy can only set whitelisted tracker constants, clamped to their bounds (server/tracker.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server"))
import tracker as T  # noqa: E402

doc = {"constants": {"LEAD": 0.5, "EDGE": 9.0, "MOVE_EVERY": -3, "SPEED": 900, "CHASE_EDGE": "x",
                     "WEAK_GATE": float("nan"), "SEARCH_AFTER": 60}}
c = T.policy_constants(doc)
assert c == {"LEAD": 0.5, "EDGE": T.POLICY_BOUNDS["EDGE"][1], "MOVE_EVERY": T.POLICY_BOUNDS["MOVE_EVERY"][0],
             "SEARCH_AFTER": 60.0}, c
assert T.policy_constants(None) == {} and T.policy_constants({"constants": None}) == {}
for k in T.POLICY_BOUNDS:
    lo, hi = T.POLICY_BOUNDS[k]
    assert lo <= getattr(T, k) <= hi, (k, getattr(T, k), "today's value must lie inside its own bounds")
print("policy ok: unknown keys dropped, values clamped, non-numbers and NaN rejected, today's constants inside the bounds")

# learned search stops: validated pairs, clamped inside the motor range, at most 10
st = T.policy_stops({"search_stops": [[1283, 530], [99999, -5], ["x", 1], [float("nan"), 3], [100, 200]] + [[1, 1]] * 20})
assert st[:3] == [(1283, 530), (3700, 40), (100, 200)] and len(st) == 10, st
assert T.policy_stops({}) == [] and T.policy_stops(None) == []

# the search visits the learned stops in that order (before the zigzag), and the saved views when there are none
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from tracker_test import World  # noqa: E402

for stops, want in (([(3000, 500), (2500, 500)], [3000, 2500]), ([], None)):
    w = World(X=1850, pet=99999)
    w.presets = lambda: [{"name": "Corridor", "x": 1019, "y": 534}, {"name": "Couch", "x": 2252, "y": 704}]
    T.SEARCH_STOPS = stops
    tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now)
    tr.search("test")
    gotos = [x for kind, x in w.moves if kind == "goto"]
    if want:
        assert gotos[:2] == want, gotos
    else:
        assert gotos[:2] == [1019, 2252], gotos
T.SEARCH_STOPS = []
print("policy ok: search stops validated, visited in the learned order, saved views when none")
