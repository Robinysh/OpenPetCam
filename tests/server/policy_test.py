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
