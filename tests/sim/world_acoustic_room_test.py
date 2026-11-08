"""The initial acoustic room from a metric range panorama: synthetic rooms with known walls.

Run: python3 tests/sim/world_acoustic_room_test.py
"""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.world.acoustic_room import build, from_world  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import os  # noqa: E402

W, H = 768, 384
FLOOR = -0.8
CEIL = 1.8


def render(polygon, yaw_deg=0.0, furniture=(), wall_scale=1.0):
    """Range panorama of a closed vertical-walled room (polygon in x,z) with a floor, a ceiling and
    optional box furniture, seen from the origin. The room is rotated by yaw_deg about y."""
    a = math.radians(yaw_deg)
    Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    P = np.asarray(polygon, float) @ Rm.T
    az = np.radians((np.arange(W) + .5) / W * 360 - 180)
    el = np.radians(90 - (np.arange(H) + .5) / H * 180)
    A, E = np.meshgrid(az, el)
    d = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
    R = np.full((H, W), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(d[..., 1] < 0, FLOOR / d[..., 1], R)
        R = np.minimum(R, np.where(d[..., 1] > 0, CEIL / d[..., 1], np.inf))
        for i in range(len(P)):
            p, q = P[i], P[(i + 1) % len(P)]
            e = q - p
            # t * (dx, dz) = p + s * e
            det = -d[..., 0] * e[1] + e[0] * d[..., 2]
            t = (-p[0] * e[1] + e[0] * p[1]) / det
            s = (d[..., 0] * p[1] - d[..., 2] * p[0]) / det
            hit = (t > 0) & (s >= 0) & (s <= 1)
            k = 1.0 - (1.0 - wall_scale) * np.clip((t - 0.8) / 1.0, 0, 1)   # learned depth: right near, short far
            R = np.where(hit & (t < R), t * k, R)
        for lo, hi in furniture:
            lo, hi = np.asarray(lo, float), np.asarray(hi, float)
            t0, t1 = (lo - 0) / d, (hi - 0) / d
            tn = np.max(np.minimum(t0, t1), -1); tf = np.min(np.maximum(t0, t1), -1)
            R = np.where((tf >= tn) & (tn > 0), np.minimum(R, tn), R)
    return R


def wall_error(doc, polygon):
    """For each true wall, the offset of the nearest parallel model wall (metres)."""
    P = np.asarray(polygon, float); errs = []
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]
        u = (q - p) / np.linalg.norm(q - p); n = np.array([-u[1], u[0]])
        best = np.inf
        for w in doc["walls"]:
            a, b = np.array(w["a"]), np.array(w["b"])
            v = (b - a) / np.linalg.norm(b - a)
            if abs(abs(v @ u) - 1) < 0.02:
                mid = (a + b) / 2
                if abs((mid - p) @ u) <= np.linalg.norm(q - p) + 0.3:
                    best = min(best, abs((a - p) @ n))
        errs.append(best)
    return np.array(errs)


class AcousticRoomTests(unittest.TestCase):
    def test_rectangle_walls_recovered_to_a_few_cm(self):
        poly = [(-2.0, -1.0), (3.0, -1.0), (3.0, 4.0), (-2.0, 4.0)]
        R = render(poly)
        doc = build(R, np.ones_like(R, bool), -FLOOR, CEIL - FLOOR)
        self.assertEqual(len(doc["walls"]), 4)
        self.assertLess(wall_error(doc, poly).max(), 0.06)
        self.assertAlmostEqual(doc["floor"]["y"], FLOOR, places=3)
        self.assertEqual(doc["ceiling"]["source"], "assumed (never imaged)")

    def test_rotated_l_shape_with_furniture(self):
        poly = [(-3.0, -1.0), (2.0, -1.0), (2.0, 1.5), (0.5, 1.5), (0.5, 4.0), (-3.0, 4.0)]
        sofa = ((-2.6, FLOOR, 2.0), (-1.8, FLOOR + 0.8, 3.6))
        R = render(poly, yaw_deg=33.0, furniture=[sofa])
        a = math.radians(33.0); Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        truth = np.asarray(poly) @ Rm.T
        doc = build(R, np.ones_like(R, bool), -FLOOR, CEIL - FLOOR)
        self.assertAlmostEqual((doc["room_axis_deg"] - 33.0 + 45) % 90 - 45, 0.0, delta=1.0)
        self.assertEqual(len(doc["walls"]), 6)
        self.assertLess(wall_error(doc, truth).max(), 0.06)

    def test_a_narrow_corridor_off_the_room_is_kept(self):
        """A 1 m corridor leaving the room, seen straight down from the camera (as the entrance corridor is), must
        survive the simplification: the jogs
        where the room narrows get their connecting walls instead of being cut across."""
        poly = [(-2.5, -0.6), (-0.5, -0.6), (-0.5, -3.0), (0.5, -3.0), (0.5, -0.6), (2.5, -0.6), (2.5, 4.0), (-2.5, 4.0)]
        R = render(poly, yaw_deg=20.0)
        a = math.radians(20.0); Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        truth = np.asarray(poly) @ Rm.T
        doc = build(R, np.ones_like(R, bool), -FLOOR, CEIL - FLOOR)
        self.assertEqual(len(doc["walls"]), 8)
        far = np.array([[0.0, -3.0]]) @ Rm.T                   # the corridor's end wall (seen straight down it) is in the model
        self.assertTrue(any(np.linalg.norm((np.array(w["a"]) + w["b"]) / 2 - far[0]) < 0.2 for w in doc["walls"]))
        self.assertLess(wall_error(doc, truth).max(), 0.08)

    def test_furniture_snaps_to_the_room_axes(self):
        poly = [(-3.0, -1.0), (2.0, -1.0), (2.0, 1.5), (0.5, 1.5), (0.5, 4.0), (-3.0, 4.0)]
        R = render(poly, yaw_deg=33.0)
        tv = {"label": "TV", "center": [0, 0, 3], "size": [1.5, 0.9, 0.05], "yaw_deg": 151.0}
        doc = build(R, np.ones_like(R, bool), -FLOOR, CEIL - FLOOR, objects=[tv])
        o = doc["objects"][0]
        self.assertEqual(o["yaw_measured_deg"], 151.0)
        self.assertAlmostEqual(((o["yaw_deg"] + doc["room_axis_deg"]) % 90 + 45) % 90 - 45, 0.0, places=1)
        self.assertLess(abs(o["yaw_deg"] - 151.0), 45.0)

    def test_walls_read_short_by_depth_are_placed_by_the_floor_junction(self):
        """Learned depth that reads walls up to 0.76x too close beyond 1.8 m (the first home): walls whose foot is in
        view come back at their true distance from the floor-wall junction."""
        poly = [(-2.0, -1.0), (3.0, -1.0), (3.0, 4.0), (-2.0, 4.0)]
        R = render(poly, wall_scale=0.76)
        doc = build(R, np.ones_like(R, bool), -FLOOR, CEIL - FLOOR)
        err = wall_error(doc, poly)
        near = [i for i, (p, q) in enumerate(zip(poly, poly[1:] + poly[:1]))
                if math.degrees(math.atan2(-FLOOR, abs(p[1]) if p[1] == q[1] else abs(p[0]))) >= 15]
        self.assertLess(err[near].max(), 0.06)                  # foot >= 15 deg below the horizon: exact geometry
        self.assertLess(doc["depth_ratio_on_walls"], 0.95)       # measured in-room: depth reads walls short
        for w in doc["walls"]:                                 # the rest: the true distance is inside the stated range
            a = np.array(w["a"])
            if w["distance_source"] == "depth":
                n = np.array([w["normal"][0], w["normal"][2]])
                true = min(abs((np.array(p) - a) @ n) + w["distance_m"] for p in poly if abs((np.array(p) - a) @ n) < 1.5)
                lo, hi = w["distance_range_m"]
                self.assertTrue(lo - 0.05 <= true <= hi + 0.1, (w["id"], lo, hi, true))

    def test_from_world_reads_the_clean_metric_range(self):
        """The pano-depth output a world build leaves: manifest + range_pano_metric.f32 + depth_evidence.npz."""
        poly = [(-2.0, -1.0), (3.0, -1.0), (3.0, 4.0), (-2.0, 4.0)]
        R = render(poly).astype(np.float32)
        with tempfile.TemporaryDirectory() as w:
            os.makedirs(os.path.join(w, "model"))
            R.tofile(os.path.join(w, "model", "range_pano_metric.f32"))
            np.savez_compressed(os.path.join(w, "model", "depth_evidence.npz"), views=np.full(R.shape, 3, np.int16))
            json.dump({"room": {"floor_height_m": -FLOOR, "ceiling_m": CEIL - FLOOR},
                       "layers": [{"id": "fused.surface", "data": {"width": W, "height": H, "metric_file": "range_pano_metric.f32", "objects": []}}]},
                      open(os.path.join(w, "model", "manifest.json"), "w"))
            doc = from_world(w)
        self.assertEqual(len(doc["walls"]), 4)
        self.assertLess(wall_error(doc, poly).max(), 0.06)
        self.assertEqual(doc["provenance"]["range"], "range_pano_metric.f32")

    def test_unimaged_directions_are_not_used(self):
        poly = [(-2.0, -1.0), (3.0, -1.0), (3.0, 4.0), (-2.0, 4.0)]
        R = render(poly)
        imaged = np.ones_like(R, bool)
        imaged[:, : W // 4] = False                      # a quarter of the azimuths never seen
        doc = build(R, imaged, -FLOOR, CEIL - FLOOR)
        self.assertGreaterEqual(len(doc["walls"]), 3)
        self.assertTrue(all(0 <= w["imaged_fraction"] <= 1 for w in doc["walls"]))


if __name__ == "__main__":
    unittest.main()
