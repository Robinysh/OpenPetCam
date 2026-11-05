"""No live A/B (owner, 2026-11-05): every hour of every day runs today's Follow constants and the learned search stops;
only an unexpired override file switches to the learned constants for a deliberate test.
  python3 tests/server/no_rotation_test.py"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import watch  # noqa: E402
import tracker as T  # noqa: E402

d = tempfile.mkdtemp()
watch.FOLLOW_POLICY = os.path.join(d, "follow_policy.json")
watch.POLICY_OVERRIDE = os.path.join(d, "policy_override")
watch.ROOM_MAP = os.path.join(d, "room_map.json")
watch.FOLLOW_PPO = os.path.join(d, "none.onnx")
json.dump({"constants": {"LEAD": 1.5}, "search_stops": [[1000, 500], [2000, 500]]}, open(watch.FOLLOW_POLICY, "w"))
lead = T.LEAD
w = watch.Watch.__new__(watch.Watch)
w.policy_arm = "base"
for h in range(0, 24 * 3):                       # three days, every hour
    w.apply_policy(1_793_750_000 + h * 3600)
    assert w.policy_arm == "base" and T.LEAD == lead and T.SEARCH_STOPS == [(1000, 500), (2000, 500)], (h, w.policy_arm)
open(watch.POLICY_OVERRIDE, "w").write(f"learned {1_793_750_000 + 7200}")
w.apply_policy(1_793_750_000 + 60)
assert w.policy_arm == "learned" and T.LEAD == 1.5
w.apply_policy(1_793_750_000 + 7300)             # expired
assert w.policy_arm == "base" and T.LEAD == lead
open(watch.POLICY_OVERRIDE, "w").write(f"ppo {1_793_750_000 + 7200}")
w.apply_policy(1_793_750_000 + 60)               # no PPO model: stays on base
assert w.policy_arm == "base"
print("no rotation ok")
