"""Room tab, server side: the server's camera model equals the room model's (sim/world/camera.py), positions come out
of the watcher's own records as the room model's presence layer computes them, and textures cannot escape the bundle.
  python3 tests/server/room_test.py"""
import json
import math
import os
import sqlite3
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, ROOT)
from sim.world.camera import Camera as SimCamera  # noqa: E402
from sim.world.geometry import floor_hit  # noqa: E402

calib = {"focal_px": 1703.0, "k1": -0.21, "k2": 0.05, "deg_per_step_pan": 0.0857, "deg_per_step_tilt": 0.0821,
         "level_tilt": 716, "roll_deg": 0.4}
d = tempfile.mkdtemp()
os.environ["PETCAM_ROOM_DIR"] = d
import roomapi  # noqa: E402
from roomcam import Camera, floor_point  # noqa: E402

# 1. one camera model: the server's rays equal the room model's for any pose, pixel and drift correction
sim, srv = SimCamera(dict(calib)), Camera(calib)
rng = np.random.default_rng(1)
for _ in range(200):
    pan, tilt = rng.uniform(0, 3700), rng.uniform(0, 1000)
    u, v = rng.uniform(0.02, 0.98, 2)
    off = tuple(rng.uniform(-6, 6, 2)) + (0.0,)
    a = sim.rays(pan, tilt, u * 1920, v * 1080, off=off)
    b = srv.ray(pan, tilt, u, v, off)
    assert np.allclose(a, b, atol=1e-9), (a, b)
print("camera: server and room model agree on 200 random rays")

# 2. positions from a test database (the watcher's schema), with a bundle in a temp room folder
json.dump({"version": 1, "calib": calib, "floor_height_m": 0.81, "ceiling_m": 2.6, "outline": [], "walls": [],
           "objects": [], "mount_offset_deg": [2.0, -4.0, 0.0], "textures": {"photo": "room_photo.jpg"}},
          open(os.path.join(d, "room.json"), "w"))
open(os.path.join(d, "room_photo.jpg"), "wb").write(b"\xff\xd8 test")
open(os.path.join(os.path.dirname(d), "secret.txt"), "w").write("outside the bundle")
con = sqlite3.connect(":memory:")
con.executescript("""CREATE TABLE obs(ts REAL, score REAL, person REAL, box TEXT, motion REAL, present INT, moving INT);
                     CREATE TABLE track(ts REAL, pan INT, tilt INT, act TEXT, weak INT);""")
t0 = 1_800_000_000.0
for k in range(0, 60, 2):                          # camera still at (1850, 600) for a minute, then moving
    con.execute("INSERT INTO track VALUES(?,?,?,?,?)", (t0 + k, 1850, 600, "", 0))
for k in range(60, 80, 2):
    con.execute("INSERT INTO track VALUES(?,?,?,?,?)", (t0 + k, 1850 + 40 * k, 600, "", 0))
box = [0.40, 0.50, 0.60, 0.80]
con.execute("INSERT INTO obs VALUES(?,?,?,?,?,?,?)", (t0 + 10, 0.9, 0, json.dumps(box), 0, 1, 0))
for k in range(12, 30, 2):                         # the same cached box re-logged: one detection, longer dwell
    con.execute("INSERT INTO obs VALUES(?,?,?,?,?,?,?)", (t0 + k, 0.9, 0, json.dumps(box), 0, 1, 0))
con.execute("INSERT INTO obs VALUES(?,?,?,?,?,?,?)", (t0 + 70, 0.9, 0, json.dumps([0.3, 0.4, 0.5, 0.7]), 0, 1, 1))  # moving camera
con.execute("INSERT INTO obs VALUES(?,?,?,?,?,?,?)", (t0 + 40, 0.9, 0, json.dumps([0.3, 0.0, 0.5, 0.05]), 0, 1, 0))  # above the horizon
rows = roomapi.positions(con, t0, t0 + 100)
assert len(rows) == 1, rows
t, x, z, moving, dwell, cut = rows[0]
want = floor_hit(sim.rays(1850, 600, 0.5 * 1920, 0.8 * 1080, off=(2.0, -4.0, 0.0)), 0.81)
assert abs(x - want[0]) < 2e-3 and abs(z - want[2]) < 2e-3, (x, z, want)
assert abs(dwell - (28 - 10 + 1.5)) < 1e-6 and cut == 0, rows[0]
print(f"positions: one detection at ({x:.2f}, {z:.2f}) m, dwell {dwell} s (cached repeats merged); moving and sky boxes dropped")

# 3. textures: only names the bundle lists, never a path out of it
assert roomapi.texture("room_photo.jpg").endswith("room_photo.jpg")
for bad in ("../secret.txt", "room.json", "/etc/passwd", ""):
    assert roomapi.texture(bad) is None, bad
# 4. no bundle: the tab says so, nothing breaks
os.remove(os.path.join(d, "room.json"))
assert roomapi.info() == {"available": False} and roomapi.positions(con, t0, t0 + 100) == []
assert floor_point(srv, 1850, 600, [0.4, 0.0, 0.6, 0.05], 0.81) is None
print("room ok")
