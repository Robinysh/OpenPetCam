"""The camera in visible light (~400-700 nm, colour CMOS behind its IR-cut filter by day).

Layers: the calibrated sweep panorama (one moment, 55 frames), an hourly composite of the minute snapshots (where the
camera happened to look each hour), and every snapshot as a timed frame the viewer projects into the room at its pose."""
import datetime as dt
import json
import os

from PIL import Image

from ..layers import Band, Layer
from ..pano import is_monochrome, stitch_best, stitch_mean
from .base import Sensor


def snapshots(ctx):
    """[(ts, pan, tilt, path)] for every minute snapshot whose pose is known and steady."""
    out = []
    for m in ctx.json("minute.json"):
        if not m.get("snap"):
            continue
        p = ctx.path("snaps", m["snap"])
        pose = ctx.pose_at(m["ts"] + 30) if os.path.exists(p) else None
        if pose:
            out.append((m["ts"] + 30, pose[0], pose[1], p))
    return out


class VisibleCamera(Sensor):
    name = "visible"
    band = Band("visible", 400e-9, 700e-9, "colour CMOS, IR-cut filter in")

    def layers(self):
        ctx, cam = self.ctx, self.ctx.camera
        out = []
        twin = ctx.twin
        poses = json.load(open(os.path.join(twin, "poses.json")))["poses"]
        off = dict(zip(cam.c.get("frames", []), cam.c.get("offsets_deg", [])))
        img, seen = stitch_best(cam, [(os.path.join(twin, p["file"]), p["pan"], p["tilt"], off.get(p["file"], (0, 0, 0)))
                                      for p in poses])
        Image.fromarray(img).save(os.path.join(ctx.model, "pano_visible_sweep.jpg"), quality=88)
        out.append(Layer("visible.sweep", self.name, self.band, "panorama", "Visible light: calibrated sweep",
                         f"{len(poses)} frames from one pan/tilt sweep, placed by the self-calibrated camera "
                         f"({cam.c.get('median_px', 0):.1f} px median reprojection) and exposure-balanced.",
                         poses[0]["t"], poses[-1]["t"],
                         {"image": "pano_visible_sweep.jpg", "seen_share": round(float(seen.mean()), 3)}))
        snaps = [s for s in snapshots(ctx) if not is_monochrome(s[3])]
        hours = {}
        for h in range(24):
            fs = [(p, pan, tilt) for t, pan, tilt, p in snaps if dt.datetime.fromtimestamp(t).hour == h]
            if len(fs) >= 3:
                im, n = stitch_mean(cam, fs)
                Image.fromarray(im).save(os.path.join(ctx.model, f"pano_visible_h{h:02d}.jpg"), quality=85)
                hours[h] = {"image": f"pano_visible_h{h:02d}.jpg", "frames": len(fs), "seen_share": round(float((n > 0).mean()), 3)}
        out.append(Layer("visible.hourly", self.name, self.band, "panorama", "Visible light by hour of day",
                         "Every daytime minute snapshot averaged by the hour it was taken, each placed at the pose the "
                         "camera held that minute. Covers only where the camera looked.",
                         snaps[0][0] if snaps else None, snaps[-1][0] if snaps else None, {"hours": hours}))
        out.append(frames_layer(ctx, "visible.frames", self.name, self.band, "Visible light: minute snapshots", snaps))
        return out


def frames_layer(ctx, lid, sensor, band, title, snaps):
    import shutil
    d = os.path.join(ctx.model, "frames")
    os.makedirs(d, exist_ok=True)
    from ..registration import at, corrections
    pano = os.path.join(ctx.model, "pano_visible_sweep.jpg")
    corr = corrections(ctx, snapshots(ctx), pano) if os.path.exists(pano) else None
    items = []
    for t, pan, tilt, p in snaps:
        rel = os.path.relpath(p, ctx.path("snaps")).replace(os.sep, "_")
        dst = os.path.join(d, rel)
        if not os.path.exists(dst):
            shutil.copyfile(p, dst)
        a, b, _c = at(corr, t)
        items.append([round(t, 1), pan, tilt, "frames/" + rel, round(a, 2), round(b, 2)])
    return Layer(lid, sensor, band, "frames", title,
                 "One 640x360 snapshot a minute from the server, with the pan/tilt from its track table at that minute "
                 "(frames taken while the motor moved are left out).",
                 items[0][0] if items else None, items[-1][0] if items else None, {"frames": items})
