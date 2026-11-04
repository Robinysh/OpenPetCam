"""'Is she moving?' when she fills most of the picture.

Regression for report 2026-11-04 17:28: she stood and walked right in front of the lens, her box was
~60% of the frame, and the app said "resting 1m". The in-box pixel change was compared with the
whole-frame average, which was mostly her own motion, so she never counted as moving. Runs locally:
  python3 tests/server/moving_test.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, "/opt/petcam")
import watch  # noqa: E402


def diff_with(box, inside, outside):
    d = np.full((90, 160), float(outside))
    x0, y0, x1, y1 = int(box[0] * 160), int(box[1] * 90), int(np.ceil(box[2] * 160)), int(np.ceil(box[3] * 90))
    d[y0:y1, x0:x1] = inside
    d[:9, :] = 0                      # the caller zeroes the clock overlay
    return d


# The report's boxes (17:28:21 -> 17:28:26): close-up, centre shifts less than 15% of her size.
big0 = (0.364, 0.003, 0.999, 0.899)
big1 = (0.384, 0.003, 0.972, 0.938)
# Legs and body changing (~9 grey levels) over a nearly still floor: frame-wide motion 4.9, as logged.
d = diff_with(big1, 9.4, 0.5)
assert 4.7 < d.mean() < 5.1, d.mean()
# The old rule (in-box mean minus whole-frame mean > 4) said "still" here; this is the bug.
x0, x1, y1 = int(big1[0] * 160), int(np.ceil(big1[2] * 160)), int(np.ceil(big1[3] * 90))
assert d[0:y1, x0:x1].mean() - d.mean() <= 4.0
assert watch.pet_moving(big0, big1, d), "close-up dog moving in front of the lens must count as moving"

# Still close-up: only sensor noise everywhere.
assert not watch.pet_moving(big0, big1, diff_with(big1, 0.7, 0.6)), "noise is not moving"
# Exposure jump: the whole picture changes by the same amount.
assert not watch.pet_moving(big0, big1, diff_with(big1, 15.0, 15.0)), "an exposure jump is not moving"

# Small dog lying still across the room, owner walking past elsewhere: not her.
small = (0.40, 0.60, 0.55, 0.80)
assert not watch.pet_moving(small, small, diff_with(small, 1.0, 12.0)), "motion elsewhere is not her"
# Small dog stretching: in-box change well above the background.
assert watch.pet_moving(small, small, diff_with(small, 9.0, 0.6)), "stretching is moving"
# Box jump between detections (walking across) with no frame diff.
assert watch.pet_moving((0.1, 0.5, 0.3, 0.8), (0.5, 0.5, 0.7, 0.8), None)
assert not watch.pet_moving(None, small, None)

# Box covering the whole picture: fall back to the frame average, no crash.
full = (0.0, 0.0, 1.0, 1.0)
assert not watch.pet_moving(None, full, diff_with(full, 6.0, 0.0))
print("moving_test: ok")
