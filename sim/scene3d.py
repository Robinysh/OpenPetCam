#!/usr/bin/env python3
"""A 3D model of the room from the camera's own sweep, computed on this computer (no cloud).

The camera only rotates, so there is no parallax for photogrammetry; depth comes from a monocular metric depth model
(Depth Anything V2, metric indoor, small; Apache-2.0) run on each sweep frame, and each frame is placed by its motor
pose. Angles: 3700 pan steps over the mount's ~355 deg gives 0.096 deg/step, which predicts a 69.4 deg wide, 41.3 deg
tall frame from the measured 723.6 x 431.1 steps per frame; a 16:9 lens at 69.4 deg is 42.3 deg tall, so the two agree.

    python3 sim/scene3d.py [eval/twin/day]   ->  eval/twin/<tag>/scene3d/{points.ply, floor.png, floor.json}

STATUS (first run, 2026-11-04): not usable yet. Per-frame monocular depth disagrees in scale between neighbouring
frames, so the fused floor is smeared (9.2 m2 of floor found, a 5 deg tilted plane, camera "0.30 m" up). Next: solve
one depth scale per frame from the ~50% overlaps (the same viewing direction must have the same depth), then fit the
floor; check the map by eye against the panorama before Find uses it.

floor.json is what Find can use: the floor plane, the camera's height above it, and a top-down grid of floor cells
(seen floor, obstacle, never seen), each with the pan/tilt that looks at it. Everything stays in eval/ (git-ignored).
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEG_PER_STEP = 355.0 / 3700
MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
STRIDE = 6            # back-project every 6th pixel of the depth map
CELL = 0.10           # floor grid, metres
MAX_DEPTH = 8.0


def rot(pan, tilt):
    """Camera-to-world rotation for a pose. World: x right (at pan 1850), y up, z forward. View right = lower pan,
    picture up = higher tilt (server/app.py aim())."""
    yaw = math.radians((1850 - pan) * DEG_PER_STEP)
    pitch = math.radians((tilt - 500) * DEG_PER_STEP)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return ry @ rx


def depth_maps(frames, pipe):
    for f in frames:
        img = Image.open(f).convert("RGB")
        out = pipe(img)
        d = out["predicted_depth"]
        d = d.squeeze().cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
        yield img, d


def main(twin=os.path.join(ROOT, "eval", "twin", "day")):
    import torch
    from transformers import pipeline
    meta = json.load(open(os.path.join(twin, "poses.json")))
    out_dir = os.path.join(twin, "scene3d")
    os.makedirs(out_dir, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline("depth-estimation", model=MODEL, device=dev)
    W, H = meta["frame_px"]
    hfov = math.radians(meta["pan_frame"] * DEG_PER_STEP)
    pts, cols, owner = [], [], []
    for i, p in enumerate(meta["poses"]):
        img, d = next(depth_maps([os.path.join(twin, p["file"])], pipe))
        dh, dw = d.shape
        fx = (dw / 2) / math.tan(hfov / 2)
        ys, xs = np.mgrid[0:dh:STRIDE, 0:dw:STRIDE]
        z = d[ys, xs]
        ok = (z > 0.2) & (z < MAX_DEPTH)
        xs, ys, z = xs[ok], ys[ok], z[ok]
        cam = np.stack([(xs - dw / 2) / fx * z, -(ys - dh / 2) / fx * z, z], 1)     # x right, y up, z forward
        world = cam @ rot(p["pan"], p["tilt"]).T
        rgb = np.asarray(img.resize((dw, dh)))[ys, xs]
        pts.append(world); cols.append(rgb); owner.append(np.full(len(world), i))
        print(f"{i + 1}/{len(meta['poses'])} pan {p['pan']} tilt {p['tilt']}: {len(world)} points, depth "
              f"{np.percentile(z, 10):.1f}-{np.percentile(z, 90):.1f} m", flush=True)
    P, C, O = np.concatenate(pts), np.concatenate(cols), np.concatenate(owner)
    # floor: the dominant horizontal plane below the camera (RANSAC on y)
    rng = np.random.default_rng(0)
    below = P[P[:, 1] < -0.2]
    best = (0, None)
    for _ in range(300):
        s = below[rng.choice(len(below), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-6:
            continue
        n = n / np.linalg.norm(n)
        if abs(n[1]) < 0.9:                    # floor is near-horizontal
            continue
        dist = np.abs((below - s[0]) @ n)
        k = int((dist < 0.04).sum())
        if k > best[0]:
            best = (k, (n if n[1] > 0 else -n, s[0]))
    n, p0 = best[1]
    height = float(abs(p0 @ n))
    up = P @ n + (-(p0 @ n))                    # height above the floor plane
    print(f"floor plane: {best[0]} inliers; camera {height:.2f} m above the floor", flush=True)
    # top-down grid on the floor plane
    e1 = np.cross(n, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    u, v = P @ e1, P @ e2
    lo_u, lo_v = np.percentile(u, 1), np.percentile(v, 1)
    hi_u, hi_v = np.percentile(u, 99), np.percentile(v, 99)
    nu, nv = int((hi_u - lo_u) / CELL) + 1, int((hi_v - lo_v) / CELL) + 1
    iu = np.clip(((u - lo_u) / CELL).astype(int), 0, nu - 1)
    iv = np.clip(((v - lo_v) / CELL).astype(int), 0, nv - 1)
    floor = np.zeros((nv, nu), int); obst = np.zeros((nv, nu), int); look = {}
    is_floor = np.abs(up) < 0.06
    is_obst = (up > 0.10) & (up < 1.2)          # furniture height (what she can hide behind)
    np.add.at(floor, (iv[is_floor], iu[is_floor]), 1)
    np.add.at(obst, (iv[is_obst], iu[is_obst]), 1)
    for k in np.flatnonzero(is_floor)[::50]:
        look.setdefault((int(iv[k]), int(iu[k])), meta["poses"][O[k]]["file"])
    img = np.full((nv, nu, 3), 235, np.uint8)
    img[floor > 2] = (120, 190, 120)
    img[obst > 4] = (200, 110, 60)
    cu, cv = int((0 - lo_u) / CELL), int((0 - lo_v) / CELL)
    img[max(0, cv - 2):cv + 3, max(0, cu - 2):cu + 3] = (30, 60, 200)
    Image.fromarray(img[::-1]).resize((nu * 4, nv * 4), Image.NEAREST).save(os.path.join(out_dir, "floor.png"))
    with open(os.path.join(out_dir, "points.ply"), "w") as f:
        sel = rng.choice(len(P), min(len(P), 400_000), replace=False)
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(sel)}\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for k in sel:
            f.write(f"{P[k,0]:.3f} {P[k,1]:.3f} {P[k,2]:.3f} {C[k,0]} {C[k,1]} {C[k,2]}\n")
    json.dump({"camera_height_m": round(height, 3), "floor_normal": [round(x, 4) for x in n], "cell_m": CELL,
               "grid": [nv, nu], "camera_cell": [cv, cu], "floor_cells": int((floor > 2).sum()),
               "obstacle_cells": int((obst > 4).sum()), "model": MODEL, "deg_per_step": DEG_PER_STEP,
               "floor_area_m2": round(float((floor > 2).sum()) * CELL * CELL, 1)},
              open(os.path.join(out_dir, "floor.json"), "w"), indent=1)
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
