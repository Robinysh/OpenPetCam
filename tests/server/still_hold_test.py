"""After a verified still sighting, an unchanged patch where she lay keeps her present while the detector misses her;
any change (she gets up, the camera moves, 30 min pass) ends the hold (reports 20261105-205458, 20261106-092218).
  python3 tests/server/still_hold_test.py
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import watch  # noqa: E402

watch.log = lambda *a: None
rng = np.random.default_rng(0)
room = rng.integers(40, 200, size=(360, 640), dtype=np.uint8)
dog = room.copy()
dog[100:200, 300:450] = 180 + rng.integers(0, 20, size=(100, 150), dtype=np.uint8)
box = [300 / 640, 100 / 360, 450 / 640, 200 / 360]


def img(a, noise=2):
    return Image.fromarray(np.clip(a.astype(int) + rng.integers(-noise, noise + 1, a.shape), 0, 255).astype(np.uint8))


w = watch.Watch.__new__(watch.Watch)
w.pose = (1200, 500)
w.camera_moved_near = lambda t, before=2.0, after=0: False
assert w.still_hold(0.0, img(dog), {"box": box}, moving=False) is None        # a sighting sets the reference
assert w.still_hold(30.0, img(dog), None, moving=False) == box               # missed, unchanged: still here
assert w.still_hold(600.0, img(dog, noise=4), None, moving=False) == box     # 10 min, sensor noise: still here
assert w.still_hold(603.0, img(room), None, moving=False) is None             # she got up: the patch changed
assert w.still_hold(606.0, img(dog), None, moving=False) is None              # and the hold stays ended
print("unchanged patch holds; a change ends it")

w.still_hold(0.0, img(dog), {"box": box}, moving=False)
w.pose = (1500, 500)
assert w.still_hold(10.0, img(dog), None, moving=False) is None               # the camera turned
w.pose = (1200, 500)
w.still_hold(0.0, img(dog), {"box": box}, moving=False)
assert w.still_hold(watch.STILL_HOLD_MAX + 1, img(dog), None, moving=False) is None   # capped
w.still_hold(0.0, img(dog), {"box": box}, moving=True)                         # a moving sighting is no reference
assert w.still_hold(5.0, img(dog), None, moving=False) is None
print("camera move, cap and moving sightings end it")
print("still hold ok")
