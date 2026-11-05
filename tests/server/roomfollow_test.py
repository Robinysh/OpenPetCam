"""Follow in metres: the geometry round-trips (a box's feet placed on the floor project back to where the box is),
a hidden floor point is one the measured room blocks from the lens, and the predicted aim leads a walking dog.
  python3 tests/server/roomfollow_test.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
from roomfollow import MetricTrack, RoomFollow  # noqa: E402

calib = {"focal_px": 1703.0, "k1": 0.0, "k2": 0.0, "deg_per_step_pan": 0.0857, "deg_per_step_tilt": 0.0821,
         "level_tilt": 716, "roll_deg": 0.0}
H, W = 180, 360
R = np.full((H, W), 8.0, np.float32)
el = 90 - (np.arange(H) + 0.5) / H * 180
az = (np.arange(W) + 0.5) / W * 360 - 180
A, E = np.meshgrid(az, el)
R[(A > 20) & (A < 40) & (E < 0)] = 1.0                    # a wall 1 m away at azimuth 20-40, below the horizon
doc = {"calib": calib, "floor_height_m": 0.8, "mount_offset_deg": [0, 0, 0]}
rf = RoomFollow(doc, R)

# 1. round trip: feet of a box -> floor -> back into the frame at the box's bottom centre
for pan, tilt, box in ((1850, 600, [0.40, 0.45, 0.60, 0.80]), (1500, 650, [0.20, 0.30, 0.35, 0.70]), (2100, 560, [0.6, 0.5, 0.8, 0.9])):
    q = rf.floor(pan, tilt, box)
    o = rf.offset(pan, tilt, np.array([q[0], -0.8, q[1]]))
    want = ((box[0] + box[2]) / 2 - 0.5, box[3] - 0.5)
    assert abs(o[0] - want[0]) < 1e-3 and abs(o[1] - want[1]) < 1e-3, (o, want)
print("round trip ok: a box's feet placed on the floor project back to the box")

# 2. hidden: behind the wall at azimuth 30 (beyond 1 m) is hidden; in front of it and elsewhere is not
import math  # noqa: E402
far = (3 * math.sin(math.radians(30)), 3 * math.cos(math.radians(30)))
near = (0.5 * math.sin(math.radians(30)), 0.5 * math.cos(math.radians(30)))
assert rf.hidden(*far) and not rf.hidden(*near) and not rf.hidden(0.0, 3.0)
print("hidden ok: blocked by the room from every pose, independent of where the camera points")

# 3. a dog walking at 0.6 m/s: the aim leads her by LEAD seconds; heading into the hidden area -> exit point
mt = MetricTrack(rf)
for k in range(8):
    x, z = -1.0 + 0.3 * k, 3.0                             # walking +x at 0.6 m/s, every 0.5 s
    mt.pts.append((k * 0.5, x, z))
vx, vz = mt.velocity()
assert abs(vx - 0.6) < 1e-6 and abs(vz) < 1e-6, (vx, vz)
qx, qz = mt.predict(3.5 + 1.0)
assert abs(qx - (1.1 + 0.6)) < 1e-6
mt2 = MetricTrack(rf)
for k in range(6):                                          # walking along azimuth 30 deg, outward; last seen 0.8 m, wall at 1 m
    r = 0.2 + 0.12 * k
    mt2.pts.append((k * 0.5, r * math.sin(math.radians(30)), r * math.cos(math.radians(30))))
a = mt2.aim(1850, 600, 2.5 + 2.0)
assert a is not None and a[2] is True and a[3] is not None, a
# where she was just seen counts as hidden -> the model is wrong there: no "hidden" action
mt3 = MetricTrack(rf)
for k in range(6):
    r = 1.5 + 0.1 * k
    mt3.pts.append((k * 0.5, r * math.sin(math.radians(30)), r * math.cos(math.radians(30))))
assert mt3.aim(1850, 600, 4.0)[2] is False
print("prediction ok: 0.6 m/s fitted, aim leads by LEAD; a path into a blocked area is flagged with its exit")
print("roomfollow ok")
