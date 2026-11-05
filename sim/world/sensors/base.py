import json
import os

import numpy as np


class Context:
    """Paths and shared state for a build: the world folder, the calibrated camera, the room model (once fused)."""

    def __init__(self, world):
        from ..camera import Camera
        self.world = os.path.abspath(world)
        self.model = os.path.join(self.world, "model")
        os.makedirs(self.model, exist_ok=True)
        self.camera = Camera(os.path.join(self.world, "calib.json"))
        self.room = None
        self._cache = {}
        tw = os.path.join(self.world, "twin.txt")             # which sweep this world is built from (default: day)
        self.twin = os.path.abspath(os.path.join(self.world, open(tw).read().strip())) if os.path.exists(tw) else \
            os.path.join(os.path.dirname(self.world), "twin", "day")

    def path(self, *p):
        return os.path.join(self.world, *p)

    def json(self, name):
        if name not in self._cache:
            p = self.path(name)
            self._cache[name] = json.load(open(p)) if os.path.exists(p) else []
        return self._cache[name]

    def pose_at(self, ts, max_gap=90):
        """The camera's pan/tilt at time ts from the server's track table, or None when unknown or moving."""
        if "track_ts" not in self._cache:
            tr = self.json("track.json")
            self._cache["track_ts"] = np.array([t["ts"] for t in tr])
        tt, tr = self._cache["track_ts"], self.json("track.json")
        if not len(tt):
            return None
        k = int(np.searchsorted(tt, ts))
        cand = [i for i in (k - 1, k) if 0 <= i < len(tt)]
        i = min(cand, key=lambda i: abs(tt[i] - ts))
        if abs(tt[i] - ts) > max_gap:
            return None
        a, b = tr[max(i - 1, 0)], tr[min(i + 1, len(tr) - 1)]
        if (a["pan"], a["tilt"]) != (b["pan"], b["tilt"]):
            return None                                   # the motor was moving around ts
        return tr[i]["pan"], tr[i]["tilt"]


class Sensor:
    name = "sensor"
    band = None
    needs_room = False        # True: runs after the room is fused (it uses the floor height)

    def __init__(self, ctx):
        self.ctx = ctx

    def collect(self):
        """Fetch raw data from the device into eval/world/raw/<name>/ (optional; the default uses what is there)."""

    def layers(self):
        return []
