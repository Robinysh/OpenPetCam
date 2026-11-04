#!/usr/bin/env python3
"""A 3D model of the room from the camera's own sweep, computed on this computer (no cloud).

The camera only rotates, so there is no parallax for photogrammetry; depth comes from a monocular metric depth model
(Depth Anything V2, metric indoor, small; Apache-2.0) run on each sweep frame, and each frame is placed by its motor
pose. Angles: 3700 pan steps over the mount's ~355 deg gives 0.096 deg/step, which predicts a 69.4 deg wide, 41.3 deg
tall frame from the measured 723.6 x 431.1 steps per frame; a 16:9 lens at 69.4 deg is 42.3 deg tall, so the two agree.

    python3 sim/scene3d.py [eval/twin/day]   ->  eval/twin/<tag>/scene3d/{points.ply, floor.png, floor.json}

STATUS (2026-11-04): not usable as a floor map yet. One depth scale per frame, solved from the ~50% overlaps (339
pairs), cuts the disagreement between overlapping frames from 38.5% to 18.2%, but the scales span 0.33-2.06: the
model errs within frames too, and 18% at 2 m is ~35 cm of height noise, so most real floor reads as an obstacle.
Next approach: segment floor pixels in each frame and place them where their viewing ray meets the floor plane (the
camera sits at one point, so a floor pixel needs no depth; the camera height is measured once). Monocular depth stays
useful for coarse geometry (walls, large furniture). Check every map by eye against the panorama before Find uses it.

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
    # 1. depth per frame (cached: the model is the slow part)
    cache = os.path.join(out_dir, "depth_cache.npz")
    poses = meta["poses"]
    if os.path.exists(cache):
        z = np.load(cache)
        depths = [z[f"d{i}"] for i in range(len(poses))]
    else:
        depths = []
        for i, p in enumerate(poses):
            _img, d = next(depth_maps([os.path.join(twin, p["file"])], pipe))
            depths.append(d.astype(np.float32))
            print(f"depth {i + 1}/{len(poses)} pan {p['pan']} tilt {p['tilt']}: {np.percentile(d, 10):.1f}-"
                  f"{np.percentile(d, 90):.1f} m", flush=True)
        np.savez_compressed(cache, **{f"d{i}": d for i, d in enumerate(depths)})
    dh, dw = depths[0].shape
    fx = (dw / 2) / math.tan(hfov / 2)
    Rs = [rot(p["pan"], p["tilt"]) for p in poses]

    def ray(xs, ys):
        v = np.stack([(xs - dw / 2) / fx, -(ys - dh / 2) / fx, np.ones_like(xs, dtype=float)], 1)
        return v                                          # camera-frame direction with z = 1 (scale by depth z)

    # 2. one depth scale per frame from the overlaps: the same viewing direction must have the same range in both frames
    ys, xs = np.mgrid[12:dh:24, 12:dw:24]
    xs, ys = xs.ravel().astype(float), ys.ravel().astype(float)
    rows, rhs, wts = [], [], []
    for i in range(len(poses)):
        for j in range(len(poses)):
            if j <= i or abs(poses[i]["pan"] - poses[j]["pan"]) > 1.2 * meta["pan_frame"] or \
                    abs(poses[i]["tilt"] - poses[j]["tilt"]) > 1.2 * meta["tilt_frame"]:
                continue
            ci = ray(xs, ys)
            zi = depths[i][ys.astype(int), xs.astype(int)]
            dirw = (ci / np.linalg.norm(ci, axis=1, keepdims=True)) @ Rs[i].T
            cj = dirw @ Rs[j]                              # into camera j
            ok = cj[:, 2] > 0.2
            u = fx * cj[:, 0] / np.where(ok, cj[:, 2], 1) + dw / 2
            v = dh / 2 - fx * cj[:, 1] / np.where(ok, cj[:, 2], 1)
            ok &= (u >= 0) & (u < dw - 1) & (v >= 0) & (v < dh - 1) & (zi > 0.2) & (zi < MAX_DEPTH)
            if ok.sum() < 40:
                continue
            zj = depths[j][v[ok].astype(int), u[ok].astype(int)]
            ri = zi[ok] * np.linalg.norm(ci[ok], axis=1)
            rj = zj * np.linalg.norm(ray(u[ok], v[ok]), axis=1)
            good = (rj > 0.2) & (rj < MAX_DEPTH)
            if good.sum() < 40:
                continue
            row = np.zeros(len(poses)); row[i], row[j] = 1, -1     # log s_i - log s_j = log r_j - log r_i
            rows.append(row); rhs.append(float(np.median(np.log(rj[good]) - np.log(ri[good])))); wts.append(math.sqrt(good.sum()))
    A = np.array(rows) * np.array(wts)[:, None]
    b = np.array(rhs) * np.array(wts)
    A = np.vstack([A, np.ones(len(poses)) * 10.0]); b = np.append(b, 0.0)     # gauge: mean log scale 0 (keep metric)
    logs = np.linalg.lstsq(A, b, rcond=None)[0]
    resid_before = float(np.median(np.abs(rhs)))
    resid_after = float(np.median(np.abs(np.array(rhs) - np.array(rows) @ logs)))
    scales = np.exp(logs)
    print(f"overlap pairs {len(rows)}: median scale disagreement {100 * resid_before:.1f}% -> {100 * resid_after:.1f}% "
          f"after one scale per frame (scales {scales.min():.2f}-{scales.max():.2f})", flush=True)

    # 3. fuse
    pts, cols, owner = [], [], []
    for i, p in enumerate(poses):
        img = Image.open(os.path.join(twin, p["file"])).convert("RGB").resize((dw, dh))
        yy, xx = np.mgrid[0:dh:STRIDE, 0:dw:STRIDE]
        z = depths[i][yy, xx] * scales[i]
        ok = (z > 0.2) & (z < MAX_DEPTH)
        xx, yy, z = xx[ok].astype(float), yy[ok].astype(float), z[ok]
        world = (ray(xx, yy) * z[:, None]) @ Rs[i].T
        pts.append(world); cols.append(np.asarray(img)[yy.astype(int), xx.astype(int)]); owner.append(np.full(len(world), i))
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
        k = int((dist < 0.05).sum())
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
    is_floor = np.abs(up) < 0.08
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
