"""Self-calibration must recover a camera it was not told about: sim/sysid_camera.py's own measurement code, run
against a virtual camera with hidden parameters (sim/virtual_camera.py), has to find them.

    python3 tests/sim/sysid_recovery_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))
import sysid_camera as S  # noqa: E402
from virtual_camera import VirtualCamera  # noqa: E402

for truth in (dict(speed=140.0, fast_ratio=3.2, start_s=0.25, frame_steps=(900.0, 520.0), backlash=60.0),
              dict(speed=216.0, fast_ratio=3.5, start_s=0.08, frame_steps=(740.0, 424.0), backlash=27.0)):
    v = VirtualCamera(**truth, seed=3)
    m = S.motor_runs(cam=v)
    assert abs(m["A"]["speed"] - truth["speed"]) <= 0.05 * truth["speed"], (m, truth)
    assert abs(m["C"]["speed"] - truth["speed"] * truth["fast_ratio"]) <= 0.08 * truth["speed"] * truth["fast_ratio"], (m, truth)
    assert abs(m["A"]["first_motion_ms"] - truth["start_s"] * 1000) <= 80, (m, truth)
    assert m["B"]["queues"] is True
    for ax, k in (("pan", 0), ("tilt", 1)):
        g = S.frame_steps(ax, cam=v, snap=v.snapshot)
        assert abs(g["steps_per_frame"] - truth["frame_steps"][k]) <= 0.03 * truth["frame_steps"][k], (ax, g, truth)
        assert abs(g["backlash_steps"] - truth["backlash"]) <= 8, (ax, g, truth)
    print(f"recovered a hidden camera: speed {m['A']['speed']} (true {truth['speed']}), start {m['A']['first_motion_ms']} ms, "
          f"frame/backlash within tolerance")
print("sysid recovery ok")
