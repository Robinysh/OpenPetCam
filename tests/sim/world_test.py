"""The world model's geometry and contracts, without any camera, server or model weights.
  1. camera.rays and camera.project invert each other (with distortion) at any pose.
  2. The world frame: pan 1850 at level tilt looks along +z; a lower pan looks right (+x); a higher tilt looks up.
  3. floor_hit puts a ray 45 deg down exactly one floor height away.
  4. Room: a wall read at a known azimuth/elevation lands where the geometry says, the outline finds it, and the
     forward model predicts the floor echo at the floor height.
  python3 tests/sim/world_test.py
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from sim.world.camera import Camera  # noqa: E402
from sim.world.geometry import dir_from, floor_hit  # noqa: E402
from sim.world.room import Room  # noqa: E402

cam = Camera({"focal_px": 1747.0, "k1": -0.24, "k2": -0.11, "deg_per_step_pan": 0.0843, "deg_per_step_tilt": 0.0803,
              "level_tilt": 728.0, "roll_deg": 0.3})
rng = np.random.default_rng(1)
for pan, tilt in [(0, 60), (1850, 728), (3600, 908), (900, 400)]:
    x, y = rng.uniform(100, 1820, 50), rng.uniform(80, 1000, 50)
    d = cam.rays(pan, tilt, x, y)
    u, v, ok = cam.project(pan, tilt, d)
    assert ok.mean() > 0.9 and np.abs(u[ok] - x[ok]).max() < 0.05 and np.abs(v[ok] - y[ok]).max() < 0.05, (pan, tilt)

c = Camera(dict(cam.c, roll_deg=0.0))
f = c.rays(1850, 728, 960, 540)
assert np.allclose(f, [0, 0, 1], atol=1e-6), f
assert c.rays(1500, 728, 960, 540)[0] > 0.3, "a lower pan must look right (+x)"
assert c.rays(1850, 900, 960, 540)[1] > 0.1, "a higher tilt must look up (+y)"

p = floor_hit(dir_from(0.0, -45.0), 0.8)
assert np.allclose(p, [0, -0.8, 0.8], atol=1e-6), p

room = Room({"surfaces": [{"label": "wall", "pts": [[-10, -math.degrees(math.atan(0.8 / 2.0))],
                                                    [10, -math.degrees(math.atan(0.8 / 2.0))]]}]}, 0.8, 0.2, {})
w = room.walls[0]
assert abs(w["distance_m"] - 2.0 * math.cos(math.radians(10))) < 0.05 or abs(w["distance_m"] - 2.0) < 0.1, w
ol = {round(a): d for a, d, _ in room.outline()}
assert ol[0] is not None and abs(ol[0] - w["distance_m"]) < 0.02 and ol[90] is None, (ol[0], ol[90])
pred = {e["label"]: e["range_m"] for e in room.predict_echoes(np.array([0, 0, 1.0]))}
assert abs(pred["floor"] - 0.8) < 1e-6 and abs(pred["countertop overhang"] - 0.2) < 1e-6 and "wall" in pred, pred
print("world model ok: rays/project invert, frame conventions, floor hit, room walls/outline/echo prediction")

# blind-spot fill: never overwrites what vision measured; fills above with the echo plane and below with the floor
from sim.world.blindfill import fill  # noqa: E402
Rt = np.full((36, 72), np.nan, np.float32); Rt[12:26, :] = 2.0                 # vision sees only a band around level
out, src = fill(Rt, 0.18, 0.8)
assert np.array_equal(out[12:26], Rt[12:26]) and (src[12:26] == 1).all()
el_top = 90 - (0 + 0.5) / 36 * 180
assert src[0, 0] == 2 and abs(out[0, 0] - 0.18 / np.sin(np.radians(el_top))) < 1e-5
assert src[-1, 0] == 3 and abs(out[-1, 0] - 0.8 / np.sin(np.radians(el_top))) < 1e-5
print("blind fill ok: vision kept, sound plane above, assumed floor below")

# every view the Room tab offers has a colour legend (owner 2026-11-05: always show the colour map)
import re as _re  # noqa: E402
_js = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server", "room.js")).read()
_views = set(_re.findall(r'\["(\w+)", "[^"]+", "[^"]*"\]', _js[_js.index("const GROUPS"):_js.index("const tx =")])) - {"heat"}
_ab = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim", "world", "app_bundle.py")).read()
_leg = set(_re.findall(r'"(\w+)": (?:gradient|swatches)\(', _ab)) | ({"radio"} if 'lg["radio"]' in _ab else set())
assert _views and _views <= _leg, f"views without a legend: {sorted(_views - _leg)}"
assert '"heat"' in _js and "showLegend" in _js
print(f"legends ok: all {len(_views)} views plus Time spent have a colour scale")
