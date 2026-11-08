"""The calibrated pan/tilt camera, for the server: where a pixel at a motor pose points, and where that ray meets the
floor. The same model as sim/world/camera.py (tests/server/room_test.py pins the two together); the calibration comes
from the room bundle (models/room/room.json), which sim/world/app_bundle.py writes from your room model.

World frame (as the room model): lens at the origin, x right, y up, z forward at pan 1850; left-handed."""
import math

import numpy as np

W, H = 1920, 1080


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Camera:
    def __init__(self, calib):
        self.c = calib
        self.f, self.k1, self.k2 = calib["focal_px"], calib["k1"], calib["k2"]

    def R(self, pan, tilt, off=(0, 0, 0)):
        c = self.c
        a, b, r = (math.radians(v) for v in off)
        yaw = math.radians((1850 - pan) * c["deg_per_step_pan"])
        pitch = -math.radians((tilt - c["level_tilt"]) * c["deg_per_step_tilt"])
        return _ry(yaw) @ _rx(pitch) @ _rz(math.radians(c["roll_deg"])) @ (_ry(b) @ _rx(a) @ _rz(r))

    def ray(self, pan, tilt, u, v, off=(0, 0, 0)):
        """Unit world direction of the point (u, v) of the frame, both 0..1 (any stream size, same field of view)."""
        xd, yd = (u * W - W / 2) / self.f, -(v * H - H / 2) / self.f
        xu, yu = xd, yd
        # Match the simulator; ten iterations left visible corner ray error.
        for _ in range(30):
            r2 = xu * xu + yu * yu
            k = 1 + self.k1 * r2 + self.k2 * r2 * r2
            xu, yu = xd / k, yd / k
        d = np.array([xu, yu, 1.0]) @ self.R(pan, tilt, off).T
        return d / np.linalg.norm(d)


def floor_point(cam, pan, tilt, box, floor_h, off=(0, 0, 0)):
    """Where a detection box's bottom centre (her feet) meets the floor, as (x, z) metres, or None above the horizon
    or farther than 12 m."""
    x0, _y0, x1, y1 = box
    d = cam.ray(pan, tilt, (x0 + x1) / 2, y1, off)
    if d[1] >= -1e-3:
        return None
    t = floor_h / -d[1]
    x, z = float(d[0] * t), float(d[2] * t)
    return (x, z) if math.hypot(x, z) <= 12 else None
