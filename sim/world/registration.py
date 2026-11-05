"""Where the camera really pointed: align every minute snapshot to the calibrated sweep panorama.

The logged pan/tilt drifts: on 2026-11-03/04 the snapshot stream sat 3-6 deg left of the sweep's frame, and tilt moved
in steps (0 deg at the sweep, -5 deg overnight, -9 deg at midday) as the motor lost or regained steps. A 5 deg tilt
error moves a dog 2 m away by half a metre. Each snapshot is projected into an equirectangular patch at its logged pose
and phase-correlated (on gradients, so infrared night frames match the daytime panorama) against the panorama; the
correction is then smoothed over an hour so a detection at any time gets the correction of its period.
Correction convention: off = (pitch_down_deg, yaw_right_deg, 0) for Camera.R(..., off)."""
import json
import os

import numpy as np

from .geometry import directions

PW, PH = 1024, 512
VERSION = 1


def measure(ctx, snaps, pano_path):
    import cv2
    cam = ctx.camera
    ppd = PW / 360
    pano = cv2.resize(cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2GRAY), (PW, PH)).astype(np.float32)
    grad = lambda a: cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    Gp = grad(pano)
    D = directions(PW, PH)
    out = []
    for t, pan, tilt, p in snaps:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = im.astype(np.float32)
        h, w = im.shape
        u, v, ok = cam.project(pan, tilt, D, w, h)
        ok &= ~((v < 0.06 * h) & (u < 0.36 * w))
        if ok.sum() < 5000:
            continue
        ys, xs = np.where(ok)
        if xs.max() - xs.min() > PW * 0.6:
            continue                                     # the patch straddles azimuth 180
        patch = np.zeros((PH, PW), np.float32)
        patch[ok] = im[v[ok].astype(int), u[ok].astype(int)]
        Y0, Y1, X0, X1 = max(0, ys.min() - 40), min(PH, ys.max() + 40), max(0, xs.min() - 40), min(PW, xs.max() + 40)
        A = grad(patch)[Y0:Y1, X0:X1] * ok[Y0:Y1, X0:X1]
        B = Gp[Y0:Y1, X0:X1]
        win = cv2.createHanningWindow(A.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(A * win, B * win)
        out.append([round(t, 1), round(dx / ppd, 2), round(dy / ppd, 2), round(float(resp), 3)])
    return out


def corrections(ctx, snaps, pano_path, refresh=False):
    """[(t, pitch_down_deg, yaw_right_deg)] smoothed over an hour; cached in raw/registration.json."""
    cache = ctx.path("raw", "registration.json")
    if os.path.exists(cache) and not refresh:
        c = json.load(open(cache))
        if c.get("version") == VERSION and c.get("pano_mtime") == os.path.getmtime(pano_path):
            return c
    m = [r for r in measure(ctx, snaps, pano_path) if r[3] > 0.08 and abs(r[1]) < 15 and abs(r[2]) < 15]
    t = np.array([r[0] for r in m])
    smooth = []
    for r in m:
        near = (np.abs(t - r[0]) < 1800)
        smooth.append([r[0], float(np.median([q[2] for q, k in zip(m, near) if k])), float(np.median([q[1] for q, k in zip(m, near) if k]))])
    c = {"version": VERSION, "pano_mtime": os.path.getmtime(pano_path), "raw": m, "smooth": smooth}
    json.dump(c, open(cache, "w"))
    return c


def at(corr, ts):
    """The correction in force at time ts (nearest smoothed sample within 2 h, else none)."""
    s = corr.get("smooth") if corr else None
    if not s:
        return (0.0, 0.0, 0.0)
    tt = np.array([x[0] for x in s])
    k = int(np.argmin(np.abs(tt - ts)))
    if abs(tt[k] - ts) > 7200:
        return (0.0, 0.0, 0.0)
    return (s[k][1], s[k][2], 0.0)
