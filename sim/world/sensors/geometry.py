"""Learned geometry from the visible sweep, run locally (Apple MPS). Two models, used for different things:

DepthAnything3Depth (Depth Anything 3, DA3-BASE, Apache-2.0): depth for all sweep frames jointly, conditioned on the
calibrated poses. The camera only rotates (no baseline), so consistency comes from the model, not from triangulation;
the metric scale comes from sound: the fitted floor plane must sit at the floor height the floor echo measured.
Scored by the same overlap test the single-image model failed (23.7% median range disagreement between frames).

MarigoldSurfaces (Marigold normals v1-1, OpenRAIL++): surface normals per frame. Pixels whose normal points up are
floor; their rays meet the floor plane at the measured height, so the floor map needs no depth at all.

Both need the separate environment ~/.venvs/da3 (torch, depth_anything_3, diffusers); run them with
    ~/.venvs/da3/bin/python -m sim.world.sensors.geometry eval/world [da3|marigold]
They write eval/world/raw/geometry/*.npz, which the build reads (layers() never needs torch)."""
import json
import math
import os
import sys

import numpy as np
from PIL import Image

from ..layers import Band, Layer
from .base import Sensor

PW, PH = 672, 378          # undistorted pinhole size handed to the models
PIN_F_FRAC = 0.5           # pinhole focal = this x calibrated focal (scaled), keeps the whole frame


def pinhole(cam):
    f = cam.f * PW / 1920 * 1.0
    return f, np.array([[f, 0, PW / 2], [0, f, PH / 2], [0, 0, 1.0]])


def undistort(cam, img):
    """Remap a 1920x1080 frame to a PW x PH pinhole image with the calibrated focal (the models expect pinhole)."""
    f, K = pinhole(cam)
    ys, xs = np.mgrid[0:PH, 0:PW].astype(float)
    x, y = (xs + 0.5 - PW / 2) / f, -(ys + 0.5 - PH / 2) / f            # camera frame, y up
    r2 = x * x + y * y
    k = 1 + cam.k1 * r2 + cam.k2 * r2 * r2
    u, v = x * k * cam.f + 960, 540 - y * k * cam.f
    a = np.asarray(img.convert("RGB"), np.float32)
    ok = (u >= 0) & (u < 1919) & (v >= 0) & (v < 1079)
    out = np.zeros((PH, PW, 3), np.uint8)
    out[ok] = a[v[ok].astype(int), u[ok].astype(int)].clip(0, 255).astype(np.uint8)
    return Image.fromarray(out), ok


def cv_extrinsic(cam, pan, tilt, off):
    """World-to-camera 4x4 in OpenCV convention (x right, y down, z forward). Our frames are y-up and left-handed;
    flipping y in both the camera and the world frame (F R F) gives a proper rotation in the right-handed convention."""
    F = np.diag([1, -1, 1.0])
    Rcw = F @ cam.R(pan, tilt, off) @ F              # camera-to-world, both in y-down
    E = np.eye(4)
    E[:3, :3] = Rcw.T
    return E


def _twin_of(world):
    tw = os.path.join(world, "twin.txt")
    return os.path.abspath(os.path.join(world, open(tw).read().strip())) if os.path.exists(tw) else \
        os.path.join(os.path.dirname(os.path.abspath(world)), "twin", "day")


def run_da3(world):
    import torch
    from depth_anything_3.api import DepthAnything3
    from ..camera import Camera
    cam = Camera(os.path.join(world, "calib.json"))
    twin = _twin_of(world)
    poses = json.load(open(os.path.join(twin, "poses.json")))["poses"]
    off = dict(zip(cam.c.get("frames", []), cam.c.get("offsets_deg", [])))
    imgs, masks, ext = [], [], []
    for p in poses:
        im, ok = undistort(cam, Image.open(os.path.join(twin, p["file"])))
        imgs.append(np.asarray(im)); masks.append(ok)
        ext.append(cv_extrinsic(cam, p["pan"], p["tilt"], off.get(p["file"], (0, 0, 0))))
    f, K = pinhole(cam)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to(dev)
    # DA3 always aligns its predicted cameras to the given ones by a similarity fit of the camera centres; ours all
    # sit at the lens (pure rotation), so that fit is degenerate. Skip it: depth stays in the model's own scale and the
    # floor echo sets metres afterwards.
    model._align_to_input_extrinsics_intrinsics = lambda ext, ixt, pred, *a, **k: pred
    pred = model.inference(image=imgs, extrinsics=np.array(ext, np.float32), intrinsics=np.array([K] * len(imgs), np.float32),
                           align_to_input_ext_scale=False, process_res=PW, process_res_method="upper_bound_resize")
    d = np.asarray(pred.depth, np.float32)
    conf = np.asarray(getattr(pred, "conf", np.ones_like(d)), np.float32)
    os.makedirs(os.path.join(world, "raw", "geometry"), exist_ok=True)
    np.savez_compressed(os.path.join(world, "raw", "geometry", "da3.npz"), depth=d, conf=conf, mask=np.array(masks),
                        files=np.array([p["file"] for p in poses]), K=K)
    print("da3 depth", d.shape, float(np.median(d)), flush=True)


def run_marigold(world):
    import torch
    from diffusers import MarigoldNormalsPipeline
    from ..camera import Camera
    cam = Camera(os.path.join(world, "calib.json"))
    twin = _twin_of(world)
    poses = json.load(open(os.path.join(twin, "poses.json")))["poses"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", torch_dtype=torch.float32).to(dev)
    N, M = [], []
    for i, p in enumerate(poses):
        im, ok = undistort(cam, Image.open(os.path.join(twin, p["file"])))
        out = pipe(im, num_inference_steps=4, ensemble_size=1, processing_resolution=PW, generator=torch.Generator().manual_seed(0))
        n = np.asarray(out.prediction[0], np.float32)          # H x W x 3, camera frame (x right, y up, z toward the viewer)
        N.append(n.astype(np.float16)); M.append(ok)
        print(f"normals {i + 1}/{len(poses)}", flush=True)
    os.makedirs(os.path.join(world, "raw", "geometry"), exist_ok=True)
    np.savez_compressed(os.path.join(world, "raw", "geometry", "marigold_normals.npz"), normals=np.array(N), mask=np.array(M),
                        files=np.array([p["file"] for p in poses]))


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    {"da3": run_da3, "marigold": run_marigold}[sys.argv[2]](sys.argv[1])


def _cam_rays_pinhole(cam):
    f, _K = pinhole(cam)
    ys, xs = np.mgrid[0:PH, 0:PW].astype(float)
    return np.stack([(xs + 0.5 - PW / 2) / f, -(ys + 0.5 - PH / 2) / f, np.ones_like(xs)], -1)   # z = 1 (planar depth)


def da3_points(ctx):
    """World points (model frame, metres after floor scaling) from the DA3 depth, plus the evaluation numbers."""
    z = np.load(ctx.path("raw", "geometry", "da3.npz"))
    D, mask, files = z["depth"], z["mask"], list(z["files"])
    cam = ctx.camera
    twin = ctx.twin
    poses = {p["file"]: p for p in json.load(open(os.path.join(twin, "poses.json")))["poses"]}
    off = dict(zip(cam.c.get("frames", []), cam.c.get("offsets_deg", [])))
    rc = _cam_rays_pinhole(cam)
    Rs = [cam.R(poses[f]["pan"], poses[f]["tilt"], off.get(f, (0, 0, 0))) for f in files]
    # overlap test: the same world direction must get the same range from two frames
    f_, _ = pinhole(cam)
    ys, xs = np.mgrid[10:PH:20, 10:PW:20]
    xs, ys = xs.ravel(), ys.ravel()
    rel = []
    for i, fi in enumerate(files):
        v = rc[ys, xs]
        ri = D[i][ys, xs] * np.linalg.norm(v, axis=1)
        w = (v / np.linalg.norm(v, axis=1, keepdims=True)) @ Rs[i].T
        for j, fj in enumerate(files):
            pi, pj = poses[fi], poses[fj]
            if j == i or abs(pi["pan"] - pj["pan"]) > 400 or abs(pi["tilt"] - pj["tilt"]) > 220:
                continue
            c = w @ Rs[j]
            ok = (c[:, 2] > 0.2) & mask[i][ys, xs]
            u = f_ * c[:, 0] / np.where(ok, c[:, 2], 1) + PW / 2
            vv = PH / 2 - f_ * c[:, 1] / np.where(ok, c[:, 2], 1)
            ok &= (u >= 0) & (u < PW - 1) & (vv >= 0) & (vv < PH - 1)
            if ok.sum() < 40:
                continue
            uj, vj = u[ok].astype(int), vv[ok].astype(int)
            ok2 = mask[j][vj, uj]
            rj = D[j][vj, uj][ok2] * np.linalg.norm(rc[vj, uj][ok2], axis=1)
            if len(rj) > 40:
                rel.append(float(np.median(np.abs(np.log(rj / ri[ok][ok2])))))
    disagreement = math.exp(float(np.median(rel))) - 1 if rel else None
    # points
    P = []
    for i, fi in enumerate(files):
        sel = mask[i][::6, ::6]
        v = rc[::6, ::6][sel]
        P.append((v * D[i][::6, ::6][sel][:, None]) @ Rs[i].T)
    P = np.concatenate(P)
    # floor: points below the lens with the most support on one horizontal plane (y = const)
    below = P[P[:, 1] < -0.05][:, 1]
    hist, edges = np.histogram(below, bins=200)
    k = int(np.argmax(hist))
    y_floor = float((edges[k] + edges[k + 1]) / 2)
    h = ctx.room.floor_height if ctx.room else 0.81
    scale = h / -y_floor
    P *= scale
    flat = float(np.median(np.abs(P[np.abs(P[:, 1] + h) < 0.25][:, 1] + h)))
    return P, {"overlap_disagreement": disagreement, "pairs": len(rel), "scale_to_metres": scale, "floor_flatness_m": flat}


class DepthAnything3Depth(Sensor):
    name = "visible.da3"
    band = Band("visible", 400e-9, 700e-9, "Depth Anything 3 (DA3-BASE, Apache-2.0) on the sweep, pose-conditioned")
    needs_room = True

    def layers(self):
        if not os.path.exists(self.ctx.path("raw", "geometry", "da3.npz")):
            return []
        P, ev = da3_points(self.ctx)
        rng = np.random.default_rng(0)
        P = P[rng.choice(len(P), min(len(P), 60000), replace=False)]
        P = P[np.linalg.norm(P, axis=1) < 9]
        # wall check: distance from each hand-placed wall line to the DA3 points near it (floor plan, wall heights)
        checks = []
        if self.ctx.room:
            pts = P[(P[:, 1] > -self.ctx.room.floor_height + 0.15)][:, [0, 2]]
            for w in self.ctx.room.walls:
                a, b = np.array(w["a"]), np.array(w["b"])
                mid = (a + b) / 2
                d = mid / (np.linalg.norm(mid) + 1e-9)
                ang = np.degrees(np.arctan2(pts[:, 0], pts[:, 1]))
                az = math.degrees(math.atan2(mid[0], mid[1]))
                sel = np.abs((ang - az + 180) % 360 - 180) < 4
                if sel.sum() > 30:
                    r = float(np.percentile(np.linalg.norm(pts[sel], axis=1), 80))
                    checks.append({"label": w["label"], "picked_m": round(float(np.linalg.norm(mid)), 2), "da3_m": round(r, 2)})
        ev["wall_checks"] = checks
        # colour each point by what the lens saw in its direction (the calibrated panorama)
        pano = os.path.join(self.ctx.model, "pano_visible_sweep.jpg")
        col = np.full((len(P), 3), 160)
        if os.path.exists(pano):
            a = np.asarray(Image.open(pano).convert("RGB"))
            d = P / np.linalg.norm(P, axis=1, keepdims=True)
            u = ((np.degrees(np.arctan2(d[:, 0], d[:, 2])) + 180) / 360 * a.shape[1]).astype(int) % a.shape[1]
            v = ((90 - np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))) / 180 * a.shape[0]).astype(int).clip(0, a.shape[0] - 1)
            col = a[v, u]
        from ..depthpano import build as build_depthpano
        dp = build_depthpano(self.ctx, ev["scale_to_metres"])
        mesh = Layer("fused.surface", "fused", Band("visible", 400e-9, 700e-9, "DA3 shape + sound scale + plane snapping, photo from the lens"),
                     "rangepano", "Fused surface: learned shape, measured scale, photo texture",
                     f"One range per direction: the median of the DA3 frames that saw it ({dp['views_median']:.0f} views "
                     f"typical, {100 * dp['frame_spread_median']:.0f}% spread between them), in metres from the sound "
                     f"floor echo; {100 * dp['floor_share']:.0f}% snapped onto the floor plane and "
                     f"{100 * dp['wall_snapped_share']:.0f}% onto modelled walls. Meshed with tears at depth jumps and "
                     "textured by the lens projection.", None, None, dp)
        return [mesh, Layer("visible.da3", self.name, self.band, "points", "Learned depth: Depth Anything 3",
                      f"All {55} sweep frames through DA3-BASE jointly with the calibrated poses; scaled so the floor sits "
                      f"at the sound-measured height. Overlapping frames disagree by "
                      f"{100 * ev['overlap_disagreement']:.1f}% (median range) where single-image Depth Anything V2 "
                      f"disagreed by 23.7%.", None, None,
                      {"columns": ["x", "y", "z", "r", "g", "b"],
                       "rows": np.hstack([np.round(P, 3), col]).tolist(), "eval": ev})]
