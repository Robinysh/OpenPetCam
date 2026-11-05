"""Where she is between detector runs, from the motion image of every frame (5 fps, no detector).

The detector and verifier run every few seconds (2.8 s median while she moves, 1 in 10 gaps over 16 s; the box's
CPU budget), and the box they give is already ~1 s old when it arrives. In between, the watcher computes a cheap
motion image on every frame anyway (160x90 grey, frame vs the one 0.2 s before). This follows that motion:

  anchor(t, box)   a verified box, timed when its frame was TAKEN. Motion frames buffered since then are replayed,
                   so the estimate catches up with the box's age at once.
  frame(t, mask)   every frame: the changed pixels within about one body of the estimate (gate = her box size)
                   move it to their centroid; too few changed pixels = she stood still, the estimate stays.
  shift(fx, fy)    our own small aim moved the picture: the estimate moves the other way, buffered frames are dropped.
  estimate(now)    (cx, cy, frames_followed, seconds_since_anchor, t_update, box_w, box_h), cx, cy in -0.5..0.5, or None
                   when there is no anchor or it is older than MAX_AGE.

Measured on 11 recorded segments (sim/world/gaptrack.py): when she walked across the picture between two
detections 8 s apart, holding the last box stayed within Follow's re-aim edge 35% of the time, this 78%.
Pure numpy (the server has no OpenCV); about 0.01 ms per frame."""
import collections
import threading

import numpy as np

W, H = 160, 90
DIFF_T = 18          # per-pixel grey change (0-255) that counts as motion
MIN_PIX = 6          # changed pixels inside the gate needed to move the estimate...
MIN_FRAC = 0.08      # ...and at least this share of her box's area: small flickers near a still dog are not her.
                     # Chosen on half the recorded segments (0, 0.03, 0.08, 0.15 tried; eval/rl/fastloop_sweep.json),
                     # reported on the other half
MAX_AGE = 12.0       # s after the last anchor the estimate is no longer trusted
BUFFER_S = 6.0       # motion frames kept for replay after a late-arriving box
OVERLAY_ROWS = 9     # the clock overlay at the top of the picture is never motion


class FastTrack:
    def __init__(self):
        self.lock = threading.Lock()
        self.buf = collections.deque()
        self.c = None                    # (cx, cy) in 0..1 of the frame
        self.size = (0.1, 0.1)
        self.t_anchor = self.t_update = None
        self.n = 0

    @staticmethod
    def mask(small, prev):
        """The motion mask from two 160x90 grey frames (int arrays), as the watcher computes its motion."""
        d = np.abs(small.astype(np.int16) - prev.astype(np.int16)) > DIFF_T
        d[:OVERLAY_ROWS, :] = False
        return d

    def _apply(self, m):
        cx, cy = self.c
        bw, bh = self.size
        gw, gh = max(bw, 0.08), max(bh, 0.10)
        x0, x1 = int(max(0, (cx - gw) * W)), int(min(W, (cx + gw) * W) + 1)
        y0, y1 = int(max(0, (cy - gh) * H)), int(min(H, (cy + gh) * H) + 1)
        sub = m[y0:y1, x0:x1]
        if int(sub.sum()) >= max(MIN_PIX, MIN_FRAC * bw * bh * W * H):
            ys, xs = np.nonzero(sub)
            self.c = (min(1.0, max(0.0, (x0 + xs.mean()) / W)), min(1.0, max(0.0, (y0 + ys.mean()) / H)))
            self.n += 1
            return True
        return False

    def frame(self, t, m, camera_moving=False):
        with self.lock:
            if camera_moving:
                self.buf.clear()         # the whole picture changed with the motor: nothing here is her
                return
            self.buf.append((t, m))
            while self.buf and t - self.buf[0][0] > BUFFER_S:
                self.buf.popleft()
            if self.c is not None and self._apply(m):
                self.t_update = t

    def anchor(self, t_capture, box):
        with self.lock:
            self.c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            self.size = (box[2] - box[0], box[3] - box[1])
            self.t_anchor, self.t_update, self.n = t_capture, t_capture, 0
            for t, m in self.buf:
                if t > t_capture and self._apply(m):
                    self.t_update = t

    def shift(self, fx, fy):
        with self.lock:
            self.buf.clear()
            if self.c is not None:
                self.c = (min(1.0, max(0.0, self.c[0] - fx)), min(1.0, max(0.0, self.c[1] - fy)))

    def drop(self):
        with self.lock:
            self.c = self.t_anchor = self.t_update = None
            self.buf.clear()

    def estimate(self, now):
        with self.lock:
            if self.c is None or now - self.t_anchor > MAX_AGE:
                return None
            return (self.c[0] - 0.5, self.c[1] - 0.5, self.n, now - self.t_anchor, self.t_update, *self.size)

    def window(self, min_side=120 / 360):
        """A square detector window (0..1 coords of a 16:9 frame) around the estimate: 2.5x her box, for detect_crop."""
        with self.lock:
            if self.c is None:
                return None
            cx, cy = self.c
            bw, bh = self.size
            side_h = min(1.0, max(min_side, 2.5 * max(bw * 16 / 9, bh)))      # in frame heights
            sw = side_h * 9 / 16
            x0 = min(max(0.0, cx - sw / 2), 1 - sw); y0 = min(max(0.0, cy - side_h / 2), 1 - side_h)
            return (x0, y0, x0 + sw, y0 + side_h)
