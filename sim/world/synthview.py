"""Synthetic camera views from the room model, and the check that they look like the real camera.

render(pan, tilt): every pixel of a W x H frame is a ray of the calibrated camera at that pose (with the mount's
drift); its colour is the room model's panorama in that direction. Because the camera only rotates, this is exact up
to calibration, drift and lighting: no parallax to fake.
check(): render the pose of real snapshots and measure how far the real picture is from the render (phase
correlation of gradients, so lighting differences do not count), in pixels of a 640 x 360 frame. A synthetic view is
used for anything (inserting the pet, testing the detector) only when this alignment holds.
    python3 -m sim.world.synthview eval/world [n]     -> eval/world/raw/synthview_check.json"""
import json
import os
import sys

import numpy as np

W, H = 640, 360


def load(world, dog_free=False):
    """dog_free: the panorama with the pet removed (dogfree.py; FileNotFoundError until it is built)."""
    from PIL import Image
    from .sensors.base import Context
    ctx = Context(world)
    name = "pano_visible_sweep_dogfree.jpg" if dog_free else "pano_visible_sweep.jpg"
    p = os.path.join(ctx.model, name)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    pano = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
    return ctx, pano


def render(ctx, pano, pan, tilt, off=(0.0, 0.0, 0.0), w=W, h=H, smooth=False):
    """smooth: bilinear between panorama pixels instead of nearest (no blocky stretch where the panorama is coarse)."""
    yy, xx = np.mgrid[0:h, 0:w]
    d = ctx.camera.rays(pan, tilt, (xx + 0.5) * 1920 / w, (yy + 0.5) * 1080 / h, off=off)
    az = np.degrees(np.arctan2(d[..., 0], d[..., 2])); el = np.degrees(np.arcsin(np.clip(d[..., 1], -1, 1)))
    PH, PW = pano.shape[:2]
    if smooth:
        u = (az + 180) / 360 * PW - 0.5; v = np.clip((90 - el) / 180 * PH - 0.5, 0, PH - 1.001)
        i0 = np.floor(u).astype(int); j0 = v.astype(int)
        fx, fy = (u - i0)[..., None], (v - j0)[..., None]
        i0 %= PW; i1 = (i0 + 1) % PW
        return (pano[j0, i0] * (1 - fx) + pano[j0, i1] * fx) * (1 - fy) + (pano[j0 + 1, i0] * (1 - fx) + pano[j0 + 1, i1] * fx) * fy
    i = ((az + 180) / 360 * PW).astype(int) % PW
    j = np.clip(((90 - el) / 180 * PH).astype(int), 0, PH - 1)
    return pano[j, i]


def _grad(a, sigma=2.0):
    from scipy.ndimage import gaussian_filter
    g = a.mean(-1) if a.ndim == 3 else a
    g = gaussian_filter(g, sigma)
    gx, gy = np.gradient(g)
    m = np.hypot(gx, gy)
    return (m - m.mean()) / (m.std() + 1e-9)


def shift_px(a, b, max_shift=40):
    """Translation between two images: normalised cross-correlation of blurred gradient magnitudes, searched within
    +-max_shift pixels (the true pose error is small; an unconstrained search locks onto panorama seams)."""
    A, B = _grad(a), _grad(b)
    r = np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))).real / A.size
    ys = np.r_[0:max_shift + 1, r.shape[0] - max_shift:r.shape[0]]
    xs = np.r_[0:max_shift + 1, r.shape[1] - max_shift:r.shape[1]]
    sub = r[np.ix_(ys, xs)]
    j, i = np.unravel_index(np.argmax(sub), sub.shape)
    dy, dx = ys[j], xs[i]
    dy = dy if dy <= max_shift else dy - r.shape[0]
    dx = dx if dx <= max_shift else dx - r.shape[1]
    return float(np.hypot(dx, dy)), float(sub.max())


def check(world, n=20):
    from PIL import Image
    from .registration import at, corrections
    from .sensors.visible import snapshots
    ctx, pano = load(world)
    snaps = [s for s in snapshots(ctx)]
    corr = corrections(ctx, snaps, os.path.join(ctx.model, "pano_visible_sweep.jpg"))
    rng = np.random.default_rng(0)
    usable = [x for x in snaps if 300 <= x[2] <= 800]              # tilts the sweep panorama covers well
    pick = [usable[i] for i in rng.choice(len(usable), min(n, len(usable)), replace=False)]
    rows = []
    for t, pan, tilt, path in pick:
        real = np.asarray(Image.open(path).convert("RGB").resize((W, H)), np.float32) / 255
        if np.ptp(real.reshape(-1, 3).mean(0)) < 0.01:            # an infrared (grey) night frame: skip, the pano is daytime
            continue
        syn = render(ctx, pano, pan, tilt, off=at(corr, t))
        s_true, peak = shift_px(real, syn)
        other = pick[int(rng.integers(len(pick)))]                  # null: a render of a different pose
        s_null, peak_null = shift_px(real, render(ctx, pano, other[1], other[2], off=at(corr, other[0])))
        rows.append({"t": t, "pan": pan, "tilt": tilt, "shift_px": round(s_true, 2), "peak": round(peak, 3), "null_shift_px": round(s_null, 2), "null_peak": round(peak_null, 3)})
    sh = [r["shift_px"] for r in rows]
    res = {"n": len(rows), "median_shift_px": round(float(np.median(sh)), 2) if sh else None,
           "p90_shift_px": round(float(np.percentile(sh, 90)), 2) if sh else None,
           "within_3px": round(float(np.mean(np.array(sh) <= 3)), 3) if sh else None,
           "null_median_shift_px": round(float(np.median([r["null_shift_px"] for r in rows])), 2) if rows else None,
           "median_peak": round(float(np.median([r["peak"] for r in rows])), 3) if rows else None,
           "null_median_peak": round(float(np.median([r["null_peak"] for r in rows])), 3) if rows else None, "rows": rows}
    json.dump(res, open(os.path.join(world, "raw", "synthview_check.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    r = check(a[0], int(a[1]) if len(a) > 1 else 20)
    print({k: v for k, v in r.items() if k != "rows"})
