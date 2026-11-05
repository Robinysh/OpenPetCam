"""Follow in the room's metres (optional; used by tracker.py when a room bundle is installed and ROOM_FOLLOW is on).

Three things the picture alone cannot give, from the room model (models/room, sim/world/app_bundle.py):
  where she will be   each verified box's bottom centre is placed on the floor (calibrated camera, pose, drift); her
                      velocity is fitted in metres, which our own camera moves do not distort, and the aim goes to
                      where she will be after the actual delay: the box's age + LEAD (the detector runs every few
                      seconds, the stream lags ~0.7 s).
  where she cannot be seen   the camera only rotates, so a floor cell is hidden from EVERY pose exactly when the
                      measured room blocks the line from the lens to it (pose-independent). If her predicted point is
                      hidden, Follow does not chase in: it points at the last visible point of her path (where she will
                      come out) and waits.
  how to frame her    her distance and real height give where her head is: the aim centres her body (feet to head),
                      not a box cut off at the frame's bottom edge.
Everything here is geometry in the server's own camera model (roomcam.py, pinned to the room model by a test)."""
import math

import numpy as np

from roomcam import W, H, Camera, floor_point

BODY_H = 0.55            # m, her height standing (a golden retriever)
HIDDEN_MARGIN = 0.92     # a floor point is hidden when the measured range in its direction is below this share


class RoomFollow:
    def __init__(self, doc, R):
        self.doc, self.R = doc, R                         # R: measured range per direction (H x W), 0 = unknown
        self.cam = Camera(doc["calib"])
        self.floor_h = float(doc["floor_height_m"])
        self.off = tuple(doc.get("mount_offset_deg") or (0, 0, 0))
        self.RH, self.RW = R.shape

    @classmethod
    def load(cls, doc, range_path):
        try:
            w, h = doc["range_shape"]
            R = np.fromfile(range_path, "<u2").astype(np.float32).reshape(h, w) / 1000.0
            return cls(doc, R)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    # ---- geometry ----
    def floor(self, pan, tilt, box):
        """Her floor point (x, z) in metres from a box at a pose, or None."""
        return floor_point(self.cam, pan, tilt, box, self.floor_h, self.off)

    def range_at(self, d):
        az = math.degrees(math.atan2(d[0], d[2])); el = math.degrees(math.asin(max(-1.0, min(1.0, d[1]))))
        i = int((az + 180) / 360 * self.RW) % self.RW
        j = min(self.RH - 1, max(0, int((90 - el) / 180 * self.RH)))
        return float(self.R[j, i])

    def hidden(self, x, z):
        """True when nothing the camera can do shows her standing here: the room blocks the line from the lens to both
        her body's centre and its top. (The floor under her is the wrong test: lying on the sofa she is plainly seen,
        while the floor beneath her is behind the sofa's front.)"""
        for hgt in (BODY_H / 2, BODY_H):
            p = np.array([x, -self.floor_h + hgt, z])
            r = float(np.linalg.norm(p))
            rr = self.range_at(p / r)
            if not (rr > 0 and rr < r * HIDDEN_MARGIN):
                return False
        return True

    def offset(self, pan, tilt, p):
        """Frame offset (fx, fy) of a world point p at a pose: +fx right, +fy down, the frame is 1.0 wide/high."""
        Rm = self.cam.R(pan, tilt, self.off)
        c = np.asarray(p, float) @ Rm
        if c[2] <= 0.05:
            return None
        x, y = c[0] / c[2], c[1] / c[2]
        return (x * self.cam.f / W, -y * self.cam.f / H)

    def body_centre(self, x, z):
        return np.array([x, -self.floor_h + BODY_H / 2, z])


class MetricTrack:
    """Her recent floor points, a velocity in metres, and the aim that keeps her whole body in view."""

    def __init__(self, rf, window=4.0):
        self.rf, self.window, self.pts = rf, window, []

    def add(self, t_capture, pan, tilt, box):
        if box[3] > 0.97:                                   # cut off at the bottom: the feet point is not her feet
            return None
        p = self.rf.floor(pan, tilt, box)
        if p is None:
            return None
        self.pts.append((t_capture, p[0], p[1]))
        while self.pts and t_capture - self.pts[0][0] > self.window:
            self.pts.pop(0)
        return p

    MIN_POINTS = 3          # a velocity from two noisy floor points is box jitter at a distance, not her walking

    def velocity(self):
        if len(self.pts) < self.MIN_POINTS or self.pts[-1][0] - self.pts[0][0] < 0.8:
            return 0.0, 0.0
        t = np.array([q[0] for q in self.pts]); x = np.array([q[1] for q in self.pts]); z = np.array([q[2] for q in self.pts])
        tc = t - t.mean()
        den = float((tc ** 2).sum()) or 1.0
        return float((tc * (x - x.mean())).sum() / den), float((tc * (z - z.mean())).sum() / den)   # least squares, m/s

    def predict(self, t):
        if not self.pts:
            return None
        t0, x0, z0 = self.pts[-1]
        vx, vz = self.velocity()
        sp = math.hypot(vx, vz)
        if sp > 2.0:                                        # faster than she runs indoors: box noise, not her
            vx, vz = vx * 2.0 / sp, vz * 2.0 / sp
        dt = max(0.0, t - t0)
        return x0 + vx * dt, z0 + vz * dt

    def aim(self, pan, tilt, t_target):
        """(fx, fy, hidden, exit_offset): the frame offset that centres her predicted body at t_target, whether that
        point is hidden from every pose, and if so the offset of the last visible point of her path."""
        q = self.predict(t_target)
        if q is None:
            return None
        # if where she was just SEEN counts as hidden, the room model is wrong there: do not act on "hidden"
        hid = self.rf.hidden(*q) and not self.rf.hidden(self.pts[-1][1], self.pts[-1][2])
        o = self.rf.offset(pan, tilt, self.rf.body_centre(*q))
        exit_o = None
        if hid:
            x0, z0 = self.pts[-1][1:]
            for k in range(20, -1, -1):                       # walk back from the prediction to the last visible point
                f = k / 20
                xe, ze = x0 + (q[0] - x0) * f, z0 + (q[1] - z0) * f
                if not self.rf.hidden(xe, ze):
                    exit_o = self.rf.offset(pan, tilt, self.rf.body_centre(xe, ze))
                    break
        return (o[0], o[1], hid, exit_o) if o else None
