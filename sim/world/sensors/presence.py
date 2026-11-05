"""Where the dog has been: a derived sensor (visible + infrared picture, the on-server detector, camera geometry).

Each detection box's bottom centre is taken as where her feet touch the floor; its ray from the calibrated camera at
that moment's pan/tilt meets the floor plane at the camera's height. Boxes cut off by the frame's bottom edge are
kept but flagged (the true contact point is below the picture, so the point is too far away)."""
import json

import numpy as np

from ..geometry import floor_hit
from ..layers import Band, Layer
from .base import Sensor


class Presence(Sensor):
    name = "presence"
    band = Band("visible", 400e-9, 900e-9, "derived from the visible and near-infrared picture")
    needs_room = True

    def layers(self):
        ctx = self.ctx
        h = ctx.room.floor_height if ctx.room else 0.85
        import os
        from ..registration import at, corrections
        from .visible import snapshots
        pano = os.path.join(ctx.model, "pano_visible_sweep.jpg")
        corr = corrections(ctx, snapshots(ctx), pano) if os.path.exists(pano) else None
        pts, last_box, repeats, last_ts = [], None, 0, None
        for o in ctx.json("obs.json"):
            if not o.get("present") or not o.get("box"):
                last_box = None
                continue
            if o["box"] == last_box:              # a cached detection re-logged, not a new observation
                repeats += 1
                if pts and o["ts"] - last_ts < 30:            # but she was still there: it extends the dwell
                    pts[-1][5] = round(o["ts"] - pts[-1][0] + 1.5, 1)
                last_ts = o["ts"]
                continue
            last_box, last_ts = o["box"], o["ts"]
            pose = ctx.pose_at(o["ts"], max_gap=5)
            if not pose:
                continue
            x0, y0, x1, y1 = json.loads(o["box"])
            d = ctx.camera.rays(pose[0], pose[1], (x0 + x1) / 2 * 1920, y1 * 1080, off=at(corr, o["ts"]))
            p = floor_hit(d, h)
            if not np.all(np.isfinite(p)) or np.hypot(p[0], p[2]) > 12:
                continue
            pts.append([round(o["ts"], 1), round(float(p[0]), 3), round(float(p[2]), 3), int(y1 > 0.97), int(o.get("moving") or 0), 1.5])
        return [Layer("presence.dog", self.name, self.band, "points", "Dog on the floor plan",
                      f"{len(pts)} detections placed on the floor ({repeats} repeated cached boxes dropped, so each counts once): box bottom centre, calibrated ray at that moment's "
                      f"pose, floor {h:.2f} m below the lens; the pose is corrected by registering each period's snapshots to the panorama. Flag 1 = box cut off at the frame edge (point too far). Dwell: how long the same detection stayed (its cached repeats), so a nap counts for its length.",
                      pts[0][0] if pts else None, pts[-1][0] if pts else None,
                      {"columns": ["t", "x", "z", "cut_off", "moving", "dwell_s"], "rows": pts})]
