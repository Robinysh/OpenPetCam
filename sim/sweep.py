#!/usr/bin/env python3
"""Sweep the camera over its whole pan/tilt range and stitch what it sees into one panorama: the background of the
pixel twin (docs/follow-learning.md). A fixed-mount camera only rotates, so every view it can produce is a crop of
this panorama (to first order; the lens sits a little off the pan axis, see sim/params.json geometry).

    python3 sim/sweep.py [--tag day|night] [--owner-present] [--pan-step 360] [--tilt-step 212]   ->  eval/twin/<tag>/NN_PAN_TILT.jpg, poses.json, panorama.jpg

Frames: 1080p main stream (prudynt snapshot channel 0), ~50% overlap, every column approached from the same pan
direction so the gear play is the same for all of them. Pictures stay in eval/ (git-ignored): they show the home.
"""
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import sysid_camera as S  # noqa: E402
from room import load_params  # noqa: E402


def snapshot_main():
    raw = S.server(f"{S.CAM} prudyntctl snapshot -c 0 -q 85", timeout=40, binary=True)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def goto(x, y):
    """Absolute pose through the app's goto() (handles the inverted tilt range check; see sysid_camera.server_goto)."""
    return S.server_goto(x, y)


def sweep(tag, pan_step=360, tilt_step=212):
    P = load_params()
    g = P["geometry"]
    out = os.path.join(ROOT, "eval", "twin", tag)
    os.makedirs(out, exist_ok=True)
    pans = list(range(0, g["pan_max"] + 1, pan_step))
    tilts = list(range(60, g["tilt_max"] - 39, tilt_step))   # app.goto keeps 40 steps off the tilt ends
    if tilts[-1] < g["tilt_max"] - 40:
        tilts.append(g["tilt_max"] - 40)                    # always reach the top: the ceiling junction
    poses, n = [], 0
    for y in tilts:
        S.hold_follow()
        goto(max(0, pans[0] - 150), y)                     # approach every column from the same (increasing) side
        for x in pans:
            S.hold_follow()
            px, py = goto(x, y)
            img = snapshot_main()
            f = f"{n:02d}_{px}_{py}.jpg"
            img.save(os.path.join(out, f), quality=90)
            poses.append({"file": f, "pan": px, "tilt": py, "t": time.time()})
            n += 1
            print(f"{n}/{len(pans) * len(tilts)} pan {px} tilt {py}", flush=True)
    json.dump({"tag": tag, "poses": poses, "frame_px": list(img.size), "pan_frame": g["pan_frame"],
               "tilt_frame": g["tilt_frame"]}, open(os.path.join(out, "poses.json"), "w"), indent=1)
    return out


def mosaic(out, scale=0.25):
    """First-pass panorama: place each frame by its motor pose (no lens model yet). Pixels per step from geometry."""
    meta = json.load(open(os.path.join(out, "poses.json")))
    fw, fh = meta["frame_px"]
    sx, sy = fw / meta["pan_frame"] * scale, fh / meta["tilt_frame"] * scale
    w = int((3700 + meta["pan_frame"]) * sx) + 2
    h = int((1000 + meta["tilt_frame"]) * sy) + 2
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w, 1), np.float32)
    tw, th = int(fw * scale), int(fh * scale)
    yy, xx = np.mgrid[0:th, 0:tw]
    feather = (np.minimum(np.minimum(xx, tw - 1 - xx), np.minimum(yy, th - 1 - yy)).astype(np.float32) + 1)[..., None]
    for p in meta["poses"]:
        im = np.asarray(Image.open(os.path.join(out, p["file"])).resize((tw, th)), np.float32)
        # view right = lower pan; image up = higher tilt, so the panorama runs pan high->low, tilt high->low
        x0 = int((3700 - p["pan"]) * sx)
        y0 = int((1000 - p["tilt"]) * sy)
        acc[y0:y0 + th, x0:x0 + tw] += im * feather
        wsum[y0:y0 + th, x0:x0 + tw] += feather
    pano = (acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)
    path = os.path.join(out, "panorama.jpg")
    Image.fromarray(pano).save(path, quality=88)
    return path


if __name__ == "__main__":
    a = sys.argv[1:]
    tag = a[a.index("--tag") + 1] if "--tag" in a else "day"
    live = json.loads(S.server("sudo cat /var/lib/petcam/live.json"))
    if S.server("sudo cat /var/lib/petcam/privacy 2>/dev/null || echo 0").strip() == "1":
        sys.exit("privacy is on")
    if live.get("present") and live.get("rest_since") and "--owner-present" not in a:
        sys.exit("the pet is resting in view: no sweep now (--owner-present if someone is with her)")
    S.hold_follow()
    start = json.loads(S.camera("motors -j"))
    try:
        ps = int(a[a.index("--pan-step") + 1]) if "--pan-step" in a else 360
        ts = int(a[a.index("--tilt-step") + 1]) if "--tilt-step" in a else 212
        out = sweep(tag, ps, ts)
        print("panorama:", mosaic(out))
    finally:
        goto(int(float(start["xpos"])), int(float(start["ypos"])))
