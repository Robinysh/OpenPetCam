"""The live A/B report must not reward losing her (review 2026-11-04). The old score divided by "home" time inferred
from the same detections, so an arm that lost her for long stretches dropped them from its own denominator: a
25%-visible candidate scored 95% against a 50%-visible base, and the verdict was KEEP. The rotation also gave each arm
the same hours every day. Here she is home the whole time; the candidate sees her half as often.
  python3 tests/sim/ab_report_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim"))
import ab_report as A  # noqa: E402

BLOCK = 7200
obs, track = [], []
t = 1_800_000_000.0 - (1_800_000_000 % 86400)          # midnight UTC
end = t + 8 * 86400
while t < end:
    arm = ["base", "learned"][(int(t // BLOCK) + int(t // 86400)) % 2]        # the watcher's shifted rotation
    if arm == "base":
        seen = (t % 120) < 60                        # short gaps: 50% visible
    else:
        seen = (t % 14400) < 3600                    # one hour seen, three hours lost: 25% visible
    obs.append((t, 1.0 if seen else 0.0))
    track.append((t, "track: ..." if seen and t % 600 < 3 else None, f"{arm}@v1", 1 if t % 600 < 3 else 0))
    t += 3.0
out, cells = A.score(obs, track)
assert abs(out["base"]["in_view"] - 0.5) < 0.02 and abs(out["learned"]["in_view"] - 0.25) < 0.03, out
v, p = A.verdict(out, cells, "learned")
assert v == "REMOVE", (v, p)
assert p["shared_hours"] >= 20, ("the shifted rotation must give both arms the same hours of the day", p)
print("half as visible -> REMOVE:", v, p)

# The same candidate but equally visible and quieter: KEEP.
obs2 = [(ts, s if arm_row[2].startswith("base") else (1.0 if (ts % 120) < 60 else 0.0)) for (ts, s), arm_row in zip(obs, track)]
track2 = [(ts, a, pol, 0 if pol.startswith("learned") else m) for ts, a, pol, m in track]
out2, cells2 = A.score(obs2, track2)
v2, p2 = A.verdict(out2, cells2, "learned")
assert v2 == "KEEP", (v2, p2, out2)
print("equally visible, fewer moves -> KEEP:", v2, p2)

# Code versions are not pooled silently: --since keeps only the named versions.
out3, _ = A.score(obs, [(ts, a, pol.replace("@v1", "@v2") if ts > obs[len(obs) // 2][0] else pol, m) for ts, a, pol, m in track],
                  since="v2")
assert out3["base"]["watched_hours"] < out["base"]["watched_hours"] * 0.6, out3
print("ab report ok")
