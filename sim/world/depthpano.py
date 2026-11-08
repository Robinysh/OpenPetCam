"""Fuse learned depth, sound and the plane model into one textured surface.

The camera only rotates about its lens, so the whole visible scene is one function: range by direction. This builds
that function as an equirectangular range panorama:
  1. every DA3 frame's depth is resampled to the directions it saw; each direction takes the median of the frames
     that saw it (the camera's frames overlap ~50%, so most directions have 2-4 independent estimates);
  2. metres: the DA3 scale that puts the floor at the sound-measured height;
  3. snapping: a direction whose range lands within 12 cm of the floor plane goes onto the plane exactly; one within
     10% of a modelled wall goes onto the wall (the planes are what sound and the panorama agree on);
  4. a 3x3 median removes speckle.
The viewer meshes it and drops triangles across depth jumps (no rubber sheets between a chair and the wall behind it),
then textures it with the same lens projection as everything else, so the photo stays sharp on the true shape."""
import json
import os

import numpy as np

from .geometry import directions

W, H = 768, 384
MERGE = "gradient"          # "gradient" (MoGe-style, default) or "median"


def guided(ctx, rng, radius=3, eps=4e-3):
    """Edge-aware smoothing of log-range, guided by the photo (He et al. guided filter): depth edges follow the edges
    in the picture, and the frame-to-frame disagreement at object borders stops looking like spikes."""
    from PIL import Image
    from scipy.ndimage import uniform_filter
    pano = os.path.join(ctx.model, "pano_visible_sweep.jpg")
    if not os.path.exists(pano):
        return rng
    I = np.asarray(Image.open(pano).convert("L").resize((W, H), Image.BILINEAR), np.float32) / 255
    ok = np.isfinite(rng) & (rng > 0)
    p = np.where(ok, np.log(np.where(ok, rng, 1)), 0).astype(np.float32)
    m = ok.astype(np.float32)
    box = lambda x: uniform_filter(x, 2 * radius + 1, mode="nearest")
    N = box(m) + 1e-6
    mI, mp = box(I * m) / N, box(p * m) / N
    cov = box(I * p * m) / N - mI * mp
    var = box(I * I * m) / N - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    q = box(a) * I + box(b)
    return np.where(ok, np.exp(q), np.nan)


def gradient_merge(est, anchor, lam=0.05):
    """Fuse per-view range maps by their log-range GRADIENTS (each view's unknown scale cancels in its own gradients),
    solved as one least-squares system with a weak pull to `anchor` so a partial panorama keeps its overall scale.
    After MoGe's merge_panorama_depth (Microsoft, MIT licence), adapted to real frames and calibrated poses."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsmr
    V, H_, W_ = est.shape
    L = np.log(np.where(est > 0, est, np.nan))
    gx_num = np.zeros((H_, W_)); gx_n = np.zeros((H_, W_))
    gy_num = np.zeros((H_ - 1, W_)); gy_n = np.zeros((H_ - 1, W_))
    for v in range(V):
        a = L[v]
        dx = a - np.roll(a, -1, 1)                     # x[i, j] - x[i, j+1], wrapping in azimuth
        okx = np.isfinite(dx)
        gx_num[okx] += dx[okx]; gx_n[okx] += 1
        dy = a[:-1] - a[1:]
        oky = np.isfinite(dy)
        gy_num[oky] += dy[oky]; gy_n[oky] += 1
    idx = np.arange(H_ * W_).reshape(H_, W_)
    rows, cols, vals, b = [], [], [], []
    r = 0
    mx = gx_n > 0
    i0, i1 = idx[mx], np.roll(idx, -1, 1)[mx]
    n = len(i0)
    rows += [np.arange(r, r + n)] * 2; cols += [i0, i1]; vals += [np.ones(n), -np.ones(n)]; b.append(gx_num[mx] / gx_n[mx]); r += n
    my = gy_n > 0
    i0, i1 = idx[:-1][my], idx[1:][my]
    n = len(i0)
    rows += [np.arange(r, r + n)] * 2; cols += [i0, i1]; vals += [np.ones(n), -np.ones(n)]; b.append(gy_num[my] / gy_n[my]); r += n
    ma = np.isfinite(anchor) & (anchor > 0)
    ia = idx[ma]
    n = len(ia)
    rows.append(np.arange(r, r + n)); cols.append(ia); vals.append(np.full(n, lam)); b.append(lam * np.log(anchor[ma])); r += n
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, H_ * W_)).tocsr()
    x0 = np.where(ma, np.log(np.where(ma, anchor, 1)), 0).ravel()
    x = lsmr(A, np.concatenate(b), x0=x0, atol=1e-6, btol=1e-6, maxiter=3000)[0]
    out = np.exp(x).reshape(H_, W_)
    return np.where(ma, out, np.nan)


def build(ctx, scale):
    from .sensors.geometry import PH, PW, pinhole
    z = np.load(ctx.path("raw", "geometry", "da3.npz"))
    D, mask, files = z["depth"], z["mask"], list(z["files"])
    cam = ctx.camera
    twin = ctx.twin
    poses = {p["file"]: p for p in json.load(open(os.path.join(twin, "poses.json")))["poses"]}
    off = dict(zip(cam.c.get("frames", []), cam.c.get("offsets_deg", [])))
    f, _K = pinhole(cam)
    dirs = directions(W, H)
    est = np.full((len(files), H, W), np.nan, np.float32)
    for i, fi in enumerate(files):
        p = poses[fi]
        c = dirs @ cam.R(p["pan"], p["tilt"], off.get(fi, (0, 0, 0)))          # into this frame's camera
        ok = c[..., 2] > 0.2
        cz = np.where(ok, c[..., 2], 1)
        u, v = f * c[..., 0] / cz + PW / 2, PH / 2 - f * c[..., 1] / cz
        # keep the inner 90% of each frame: depth models are least reliable at the borders
        ok &= (u >= 0.05 * PW) & (u < 0.95 * PW) & (v >= 0.05 * PH) & (v < 0.95 * PH)
        ui, vi = u.astype(int).clip(0, PW - 1), v.astype(int).clip(0, PH - 1)
        ok &= mask[i][vi, ui]
        r = D[i][vi, ui] / cz * scale                                          # planar depth -> range, metres
        est[i][ok] = r[ok]
    n = np.sum(~np.isnan(est), 0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rng_median = np.nanmedian(est, 0)
        spread = np.nanstd(est, 0) / np.maximum(rng_median, 1e-3)
    rng = gradient_merge(np.nan_to_num(est, nan=0.0), rng_median) if MERGE == "gradient" else rng_median
    ctx._cache["depthpano_median"] = rng_median
    h = ctx.room.floor_height
    ctx._cache["depthpano_raw"] = rng.copy()
    ctx._cache["depthpano_views"] = n
    ctx._cache["depthpano_spread"] = spread
    np.savez_compressed(os.path.join(ctx.model, "depth_evidence.npz"), views=n.astype(np.int16), spread=spread.astype(np.float32))
    rng = guided(ctx, rng)
    # the clean metric range: learned depth in metres, before any snapping to modelled planes or object cuts.
    # The initial acoustic room (acoustic_room.py) is built from this, so it never inherits a modelled wall.
    np.where(np.isfinite(rng) & (rng > 0.05) & (rng < 60), rng, 0).astype(np.float32).tofile(os.path.join(ctx.model, "range_pano_metric.f32"))
    # snap: floor
    dy = dirs[..., 1]
    floor_r = np.where(dy < -0.02, h / -np.where(dy < -0.02, dy, -1), np.inf)
    on_floor = np.isfinite(rng) & (np.abs(rng * dy + h) < 0.12) & np.isfinite(floor_r)
    out = np.where(on_floor, floor_r, rng)
    # snap: walls (ray / vertical plane through the wall segment)
    snapped = np.zeros_like(on_floor)
    for w in ctx.room.walls:
        a, b = np.array(w["a"]), np.array(w["b"])
        e = b - a
        nrm = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        dx, dz = dirs[..., 0], dirs[..., 2]
        den = dx * nrm[0] + dz * nrm[1]
        t = np.where(np.abs(den) > 1e-3, (a @ nrm) / np.where(np.abs(den) > 1e-3, den, 1), np.nan)
        s = ((dx * t - a[0]) * e[0] + (dz * t - a[1]) * e[1]) / (e @ e)
        hy = dy * t
        hit = (t > 0) & (s > -0.02) & (s < 1.02) & (hy > -h) & (hy < 2.6 - h)
        near = hit & np.isfinite(rng) & (np.abs(t - rng) < 0.10 * rng) & ~on_floor
        out = np.where(near, t, out)
        snapped |= near
    # speckle: 3x3 median on the unsnapped part
    from scipy.ndimage import generic_filter
    free = np.isfinite(out) & ~on_floor & ~snapped
    med = generic_filter(np.where(np.isfinite(out), out, np.nan), np.nanmedian, size=3, mode="nearest")
    out = np.where(free & np.isfinite(med), med, out)
    out = np.where(np.isfinite(out) & (out < 12), out, 0).astype(np.float32)   # 0 = no estimate
    out.tofile(os.path.join(ctx.model, "range_pano_full.f32"))                  # before the boxes cut it (objects refit)
    from .objects import build as build_objects
    objects, cut = build_objects(ctx, out, W, H)
    out = np.where(cut, 0, out).astype(np.float32)                              # objects replace the mesh there
    # sightline carving: the camera saw the dog's feet along these rays, so nothing stands in front of her there
    from .objects import dog_floor_points
    carved = 0
    for x, zz in dog_floor_points(ctx):
        rh = float(np.hypot(x, zz))
        if rh < 0.2:
            continue
        az = np.degrees(np.arctan2(x, zz)); el = -np.degrees(np.arctan2(h, rh))
        i, j = int((az + 180) / 360 * W) % W, min(H - 1, int((90 - el) / 180 * H))
        slant = float(np.hypot(rh, h))
        blk = out[max(0, j - 1):j + 2, max(0, i - 1):i + 2]
        bad = (blk > 0) & (blk < slant / 1.15)
        if bad.any():
            blk[bad] = slant
            carved += int(bad.sum())
    path = os.path.join(ctx.model, "range_pano.f32")
    out.tofile(path)
    seen = out > 0
    stats = {"width": W, "height": H, "file": "range_pano.f32", "metric_file": "range_pano_metric.f32", "seen_share": round(float(seen.mean()), 3),
             "floor_share": round(float((on_floor & seen).sum() / max(seen.sum(), 1)), 3),
             "wall_snapped_share": round(float((snapped & seen).sum() / max(seen.sum(), 1)), 3),
             "views_median": float(np.median(n[seen])) if seen.any() else 0,
             "frame_spread_median": round(float(np.nanmedian(spread[seen & (n > 1)])), 3), "objects": objects,
             "sightline_carved_pixels": carved}
    return stats
