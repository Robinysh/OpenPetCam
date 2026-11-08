"""A panorama of the room without the pet in it, for synthetic camera views.

The sweep panorama (pano_visible_sweep.jpg) is one moment, and the pet may have been lying in view during it: she is
then part of the 'room', and every synthetic view of that corner shows her. Here every sweep frame is judged by the
server's detector and pet verifier; each verified box (grown by GROW of its size each side, because the box is often
cut by the frame edge or misses her tail) is mapped to the directions it covers. She lies still during a sweep, so the
union of those directions is where she was; every frame taken while she was there (WINDOW_S around her sightings)
is barred from supplying them, and the other frames that see those directions (taken at another time) fill them.
The patch is colour-matched to the panorama in a ring around her region (a smooth gain field) and blended in over a
soft edge. Directions of her region no other frame sees stay as the sweep had them, and are counted
(region_left_to_sweep_frames): the region is grown generously, so that is mostly wall above her.
Check: render the poses of the frames in which she was verified, from the old panorama and from this one, and judge
them: the old renders should show her (the check can see her), the new ones should not.
Seams are feathered and sampling is bilinear (pano.stitch_blend), so exposure steps between frames fade.
    python3 -m sim.world.dogfree eval/world   -> eval/world/model/pano_visible_sweep_dogfree.jpg, raw/dogfree.json"""
import json
import os
import sys

import numpy as np

GROW = 0.35
FEATHER = 0.06          # seam feather (pano.stitch_blend)
RING_PX, EDGE_PX = 30, 40   # colour-match ring and soft edge of the patch, panorama pixels
GAIN_SIGMA = 25.0
WINDOW_S = 300.0
PW, PH = 2048, 1024


def main(world):
    from PIL import Image
    from scipy import ndimage
    from .geometry import directions
    from .pano import stitch_blend
    from .sensors.base import Context
    from .synthdog import load_judge
    from . import synthview
    ctx = Context(world); cam = ctx.camera
    poses = json.load(open(os.path.join(ctx.twin, "poses.json")))["poses"]
    off = dict(zip(cam.c.get("frames", []), cam.c.get("offsets_deg", [])))
    frames = [(os.path.join(ctx.twin, p["file"]), p["pan"], p["tilt"], tuple(off.get(p["file"], (0, 0, 0)))) for p in poses]
    judge = load_judge()
    D = directions(PW, PH).astype(np.float32)
    dog = np.zeros((PH, PW), bool)
    found = []
    for k, (path, pan, tilt, o) in enumerate(frames):
        small = np.asarray(Image.open(path).convert("RGB").resize((640, 360)))
        for p, b in judge(small):
            w, h = b[2] - b[0], b[3] - b[1]
            g = [max(0, b[0] - GROW * w), max(0, b[1] - GROW * h), min(1, b[2] + GROW * w), min(1, b[3] + GROW * h)]
            u, v, ok = cam.project(pan, tilt, D, 1920, 1080, off=o)
            dog |= ok & (u >= g[0] * 1920) & (u <= g[2] * 1920) & (v >= g[1] * 1080) & (v <= g[3] * 1080)
            found.append({"frame": poses[k]["file"], "t": poses[k]["t"], "p": round(p, 3), "box": [round(x, 3) for x in b]})
    ts = [f["t"] for f in found]
    barred = [k for k, p in enumerate(poses) if ts and min(ts) - WINDOW_S <= p["t"] <= max(ts) + WINDOW_S]
    # base: every frame, feathered; patch: only the frames taken while she was elsewhere (another time)
    base, seen, _ = stitch_blend(cam, frames, PW, PH, feather=FEATHER)
    keep = [k for k in range(len(frames)) if k not in barred]
    patch, pseen, _ = stitch_blend(cam, [frames[k] for k in keep], PW, PH, feather=FEATHER)
    # (the hourly composites were tried as a fill where no other frame sees her region: they are 640x360 means of
    # differently posed snapshots and look blocky, so that part is left to the original frames and counted instead)
    # the patch was exposed differently: match its colour to the base in a ring just outside her region, then blend it
    # in over a soft edge so no rectangle shows
    dist_out = ndimage.distance_transform_edt(~dog)
    ring = (dist_out > 2) & (dist_out <= RING_PX) & seen & pseen
    gain = (np.median(base[ring].astype(np.float32), 0) + 1) / (np.median(patch[ring].astype(np.float32), 0) + 1) if ring.sum() > 50 else np.ones(3)
    # a smooth gain field (normalised convolution of the ring's ratio), so a wall and a floor in the patch each match
    rf = ring.astype(np.float32)
    num = np.stack([ndimage.gaussian_filter(base[..., c] * rf, GAIN_SIGMA) for c in range(3)], -1)
    den = np.stack([ndimage.gaussian_filter(patch[..., c] * rf, GAIN_SIGMA) for c in range(3)], -1)
    wt = ndimage.gaussian_filter(rf, GAIN_SIGMA)[..., None]
    k = np.clip(wt / (wt.max() * 0.02 + 1e-9), 0, 1)                # trust the field where the ring is near
    field = np.clip(np.where(den > 1e-3, (num + 1e-3) / (den + 1e-3), gain), 0.75, 1.4) * k + gain * (1 - k)
    dist_in_patch = ndimage.distance_transform_edt(pseen)          # fade out before the patch frames' own edge too
    alpha = np.clip(1 - (dist_out - 2) / EDGE_PX, 0, 1) * np.clip(dist_in_patch / (EDGE_PX / 2), 0, 1)
    img = (patch.astype(np.float32) * field * alpha[..., None] + base.astype(np.float32) * (1 - alpha[..., None])).clip(0, 255).astype(np.uint8)
    src = np.where(dog & (alpha > 0.5), 1, np.where(dog, 0, 3))      # 1: her region from other frames, 0: still the old
    out = os.path.join(ctx.model, "pano_visible_sweep_dogfree.jpg")
    Image.fromarray(img).save(out, quality=92)
    # check: renders at the poses where she was verified, old panorama vs this one
    old = synthview.load(world)[1]
    new = img.astype(np.float32) / 255
    chk = []
    for f in found:
        p = next(q for q in poses if q["file"] == f["frame"])
        o = off.get(p["file"], (0, 0, 0))
        a = (synthview.render(ctx, old, p["pan"], p["tilt"], off=o, smooth=True) * 255).astype(np.uint8)
        b = (synthview.render(ctx, new, p["pan"], p["tilt"], off=o, smooth=True) * 255).astype(np.uint8)
        chk.append({"frame": f["frame"], "old_render_verified": bool(judge(a)), "dogfree_render_verified": bool(judge(b))})
    res = {"frames": len(frames), "sightings": found, "barred_frames": len(barred), "region_share": round(float(dog.mean()), 4),
           "region_filled_by_other_frames": round(float((src[dog] == 1).mean()), 3) if dog.any() else None,
           "region_left_to_sweep_frames": round(float((src[dog] == 0).mean()), 3) if dog.any() else None,
           "patch_gain_rgb": [round(float(g), 3) for g in gain],
           "seen_share": round(float(seen.mean()), 3), "check": chk}
    json.dump(res, open(os.path.join(ctx.world, "raw", "dogfree.json"), "w"), indent=1)
    np.save(os.path.join(ctx.world, "raw", "dogfree_region.npy"), dog)
    return res


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=1))
