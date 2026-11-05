"""Stitching frames into equirectangular panoramas with the calibrated camera."""
import numpy as np
from PIL import Image

from .geometry import directions


def _sample(a, u, v):
    return a[np.clip(v.astype(int), 0, a.shape[0] - 1), np.clip(u.astype(int), 0, a.shape[1] - 1)]


def _overlay(u, v, w, h):
    return (v < 0.05 * h) & (u < 0.36 * w)          # the camera's timestamp, top left


def stitch_best(cam, frames, pw=2048, ph=1024, balance=True):
    """frames: [(image path, pan, tilt, pose offset)]. Each direction takes the frame that sees it nearest its centre.
    balance: one RGB gain per frame, solved so overlapping frames agree (the camera re-exposes every frame)."""
    D = directions(pw, ph)
    proj = []
    for path, pan, tilt, off in frames:
        a = np.asarray(Image.open(path).convert("RGB"), np.float32)
        h, w = a.shape[:2]
        u, v, ok = cam.project(pan, tilt, D, w, h, off=off)
        ok &= ~_overlay(u, v, w, h)
        r = np.where(ok, np.hypot(u / w - 0.5, (v / h - 0.5) * h / w), 9)
        proj.append((a, u, v, ok, r))
    n = len(proj)
    gains = np.ones((n, 3))
    if balance and n > 1:
        rows, rhs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                both = proj[i][3] & proj[j][3]
                if both.sum() < 2000:
                    continue
                ci = _sample(proj[i][0], proj[i][1][both], proj[i][2][both]).mean(0) + 1
                cj = _sample(proj[j][0], proj[j][1][both], proj[j][2][both]).mean(0) + 1
                row = np.zeros(n); row[i], row[j] = 1, -1
                rows.append(row); rhs.append(np.log(cj / ci))        # log g_i - log g_j = log(c_j / c_i)
        if rows:
            A = np.vstack(rows + [np.ones(n) * 5]); b = np.vstack(rhs + [np.zeros(3)])
            gains = np.exp(np.linalg.lstsq(A, b, rcond=None)[0])
    out = np.zeros((ph, pw, 3), np.float32)
    best = np.full((ph, pw), 9.0)
    for k, (a, u, v, ok, r) in enumerate(proj):
        take = ok & (r < best)
        out[take] = _sample(a, u[take], v[take]) * gains[k]
        best[take] = r[take]
    return out.clip(0, 255).astype(np.uint8), best < 9


def stitch_mean(cam, frames, pw=1024, ph=512):
    """frames: [(image path, pan, tilt)]. Mean of every frame that saw each direction; also returns the count."""
    D = directions(pw, ph)
    acc = np.zeros((ph, pw, 3)); n = np.zeros((ph, pw))
    for path, pan, tilt in frames:
        a = np.asarray(Image.open(path).convert("RGB"), np.float32)
        h, w = a.shape[:2]
        u, v, ok = cam.project(pan, tilt, D, w, h)
        ok &= ~_overlay(u, v, w, h)
        acc[ok] += _sample(a, u[ok], v[ok]); n[ok] += 1
    with np.errstate(invalid="ignore"):
        img = np.nan_to_num(acc / n[..., None])
    return img.clip(0, 255).astype(np.uint8), n


def is_monochrome(path):
    a = np.asarray(Image.open(path).convert("RGB").resize((64, 36)), np.float32)
    return float(np.mean(np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2]))) < 4.0
