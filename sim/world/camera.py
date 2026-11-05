"""The camera as a measuring instrument: a rotation-only pan/tilt camera calibrated from its own sweep.

World frame: the lens at the origin, y up, z forward at pan 1850 (yaw 0), x right. Left-handed (like a camera with y flipped up): renderers that are right-handed (three.js, Blender) must mirror one axis. View right = lower pan, picture up
= higher tilt. calibrate() fits focal length, two radial distortion terms, degrees per motor step (pan, tilt), the
level tilt, roll and a per-frame pose correction (backlash, end stops) from SIFT matches plus dense edge matches between
overlapping frames (white walls give SIFT almost nothing).
On the 2026-11-04 day sweep it reprojects to 1.8 px (median) and measured 0.084 deg/step pan, 0.080 tilt, level at
tilt 728: the earlier assumption (0.096 deg/step, level at 500, tilt sign inverted) was wrong.

    python3 sim/world/camera.py eval/twin/day eval/world/calib.json
"""
import json
import math
import os
import sys

import numpy as np


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Camera:
    W, H = 1920, 1080

    def __init__(self, calib):
        self.c = calib if isinstance(calib, dict) else json.load(open(calib))
        self.f, self.k1, self.k2 = self.c["focal_px"], self.c["k1"], self.c["k2"]
        self.frame_off = {}

    def yaw(self, pan):
        return math.radians((1850 - pan) * self.c["deg_per_step_pan"])

    def pitch(self, tilt):
        return -math.radians((tilt - self.c["level_tilt"]) * self.c["deg_per_step_tilt"])

    def R(self, pan, tilt, off=(0, 0, 0)):
        """Camera-to-world rotation (columns: camera x, y, z in world)."""
        a, b, c = np.radians(off)
        return _ry(self.yaw(pan)) @ _rx(self.pitch(tilt)) @ _rz(math.radians(self.c["roll_deg"])) @ (_ry(b) @ _rx(a) @ _rz(c))

    def rays(self, pan, tilt, x, y, w=None, h=None, off=(0, 0, 0)):
        """Unit world directions for pixel coordinates x, y of a w x h image (any size with the same field of view)."""
        w, h = w or self.W, h or self.H
        s = self.W / w
        x, y = np.asarray(x, float) * s, np.asarray(y, float) * s
        xd, yd = (x - self.W / 2) / self.f, -(y - self.H / 2) / self.f
        xu, yu = xd.copy(), yd.copy()
        for _ in range(10):
            r2 = xu * xu + yu * yu
            k = 1 + self.k1 * r2 + self.k2 * r2 * r2
            xu, yu = xd / k, yd / k
        v = np.stack([xu, yu, np.ones_like(xu)], -1) @ self.R(pan, tilt, off).T
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def project(self, pan, tilt, d, w=None, h=None, off=(0, 0, 0)):
        """World directions -> pixel coordinates (u, v) and a mask of directions inside the image."""
        w, h = w or self.W, h or self.H
        c = np.asarray(d) @ self.R(pan, tilt, off)
        front = c[..., 2] > 0.15
        z = np.where(front, c[..., 2], 1)
        x, y = c[..., 0] / z, c[..., 1] / z
        r2 = x * x + y * y
        k = 1 + self.k1 * r2 + self.k2 * r2 * r2
        u = (x * k * self.f + self.W / 2) * w / self.W
        v = (self.H / 2 - y * k * self.f) * w / self.W
        ok = front & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) & (r2 < 0.45)
        return u, v, ok


def _features(P, twin, S):
    import cv2
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    sift = cv2.SIFT_create(6000)
    imgs, kd = [], []
    for p in P:
        g = cv2.resize(cv2.imread(os.path.join(twin, p["file"]), cv2.IMREAD_GRAYSCALE), None, fx=S, fy=S)
        g[:20, :350] = 0                       # timestamp overlay
        g = clahe.apply(g)                     # white walls: bring out the faint edges
        imgs.append(g)
        kd.append(sift.detectAndCompute(g, None))
    return imgs, kd


def _overlapping(P):
    return [(i, j) for i in range(len(P)) for j in range(i + 1, len(P))
            if abs(P[i]["pan"] - P[j]["pan"]) <= 400 and abs(P[i]["tilt"] - P[j]["tilt"]) <= 220]


def _sift_pairs(P, kd, S):
    import cv2
    bf = cv2.BFMatcher()
    pairs = []
    for i, j in _overlapping(P):
        if kd[i][1] is None or kd[j][1] is None:
            continue
        mm = [a for a, b in bf.knnMatch(kd[i][1], kd[j][1], k=2) if a.distance < 0.8 * b.distance]
        if len(mm) < 15:
            continue
        a = np.float32([kd[i][0][x.queryIdx].pt for x in mm])
        b = np.float32([kd[j][0][x.trainIdx].pt for x in mm])
        _h, inl = cv2.findHomography(a, b, cv2.RANSAC, 4.0)
        if inl is not None and inl.sum() >= 12:
            inl = inl.ravel().astype(bool)
            pairs.append((i, j, a[inl] / S, b[inl] / S))
    return pairs


def _dense_pairs(cam, P, imgs, S, off):
    """Edges on textureless surfaces: for a grid of points in frame i, predict where they fall in frame j with the
    current model, then refine by template matching on the gradient image (search +-24 px). Keeps confident matches."""
    import cv2
    grad = [cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) for g in imgs]
    h, w = imgs[0].shape
    T, R = 12, 24                                   # template half-size, search radius (half-res px)
    ys, xs = np.mgrid[40:h - 40:28, 40:w - 40:28]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    pairs = []
    for i, j in _overlapping(P):
        gi, gj = grad[i], grad[j]
        d = cam.rays(P[i]["pan"], P[i]["tilt"], pts[:, 0] / S, pts[:, 1] / S, off=off[i])
        u, v, ok = cam.project(P[j]["pan"], P[j]["tilt"], d, off=off[j])
        u, v = u * S, v * S
        A, B = [], []
        for (x, y), uu, vv, k in zip(pts, u, v, ok):
            if not k or not (T + R <= uu < w - T - R and T + R <= vv < h - T - R):
                continue
            tpl = gi[int(y) - T:int(y) + T + 1, int(x) - T:int(x) + T + 1]
            if tpl.std() < 6:                           # nothing to lock onto
                continue
            win = gj[int(vv) - T - R:int(vv) + T + R + 1, int(uu) - T - R:int(uu) + T + R + 1]
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _mn, mx, _l, loc = cv2.minMaxLoc(res)
            if mx < 0.75:
                continue
            A.append((x, y)); B.append((int(uu) - R + loc[0], int(vv) - R + loc[1]))
        if len(A) >= 8:
            pairs.append((i, j, np.float32(A) / S, np.float32(B) / S))
    return pairs


def calibrate(twin, out, prior_deg=3.0):
    """Fit the camera from a sweep folder (poses.json + frames). Needs opencv and scipy.
    Two passes: SIFT matches (contrast-enhanced, so faint edges count), then dense edge matches predicted by the
    first fit, which constrain frames of white walls that SIFT cannot. prior_deg: how far a frame may sit from where
    the motor says it is before it costs as much as one pixel of misfit (end stops and backlash need room)."""
    from scipy.optimize import least_squares
    P = json.load(open(os.path.join(twin, "poses.json")))["poses"]
    S = 0.5
    imgs, kd = _features(P, twin, S)
    lg = os.path.join(os.path.dirname(os.path.abspath(out)), "raw", "matches_lightglue.npz")
    if os.path.exists(lg):                             # ALIKED + LightGlue (sim/world/matches.py): many more on white walls
        import cv2
        z = np.load(lg)
        pairs = []
        for key in z.files:
            i, j = map(int, key.split("_"))
            m = z[key].astype(np.float32)
            _h, inl = cv2.findHomography(m[:, :2] * S, m[:, 2:] * S, cv2.RANSAC, 4.0)
            if inl is not None and inl.sum() >= 12:
                inl = inl.ravel().astype(bool)
                pairs.append((i, j, m[inl, :2], m[inl, 2:]))
        matcher = "ALIKED+LightGlue"
    else:
        pairs = _sift_pairs(P, kd, S)
        matcher = "SIFT"
    n = len(P)

    def unpack(th):
        return ({"focal_px": th[0], "k1": th[1], "k2": th[2], "deg_per_step_pan": th[3], "deg_per_step_tilt": th[4],
                 "level_tilt": th[5], "roll_deg": th[6]}, th[7:].reshape(n, 3))

    def solve(pairs, th0):
        def res(th):
            c, off = unpack(th)
            cam = Camera(c)
            out = []
            for i, j, a, b in pairs:
                d = cam.rays(P[i]["pan"], P[i]["tilt"], a[:, 0], a[:, 1], off=off[i])
                u, v, ok = cam.project(P[j]["pan"], P[j]["tilt"], d, off=off[j])
                out.append((np.where(ok, np.stack([u, v], 1).T, b.T).T - b).ravel() * S)
            out.append(off.ravel() / prior_deg)
            return np.concatenate(out)
        xs = np.concatenate([[100, 0.05, 0.05, 0.002, 0.002, 30, 1], np.ones(3 * n)])
        r = least_squares(res, th0, loss="huber", f_scale=3.0, x_scale=xs, max_nfev=300)
        return r, res

    th0 = np.concatenate([[1747, -0.24, -0.11, 0.0843, 0.0803, 728, 0.3], np.zeros(3 * n)])
    r, res = solve(pairs, th0)
    c, off = unpack(r.x)
    def pair_errors(c, off, pp):
        cam = Camera(c)
        per = []
        for i, j, a, b in pp:
            d = cam.rays(P[i]["pan"], P[i]["tilt"], a[:, 0], a[:, 1], off=off[i])
            u, v, ok = cam.project(P[j]["pan"], P[j]["tilt"], d, off=off[j])
            per.append(float(np.median(np.hypot(u - b[:, 0], v - b[:, 1])[ok])) if ok.any() else 999.0)
        return per

    # SIFT pairs that the first fit cannot explain are wrong matches (repeating doors, tiles): drop and refit
    keep = [p_ for p_, e in zip(pairs, pair_errors(c, off, pairs)) if e < 12]
    dropped_sift = len(pairs) - len(keep)
    r, res = solve(keep, r.x)
    c, off = unpack(r.x)
    dense = _dense_pairs(Camera(c), P, imgs, S, off)
    dense = [p_ for p_, e in zip(dense, pair_errors(c, off, dense)) if e < 6]     # dense matches only refine
    allp = keep + dense
    r, res = solve(allp, r.x)
    c, off = unpack(r.x)
    per = pair_errors(c, off, allp)
    c = {k: float(v) for k, v in c.items()}
    c["offsets_deg"] = off.tolist()
    c["frames"] = [p["file"] for p in P]
    c["median_px"] = float(np.median(per))
    c["worst_pairs"] = sorted(([P[i]["file"], P[j]["file"], round(e, 1)] for (i, j, _a, _b), e in zip(allp, per)),
                              key=lambda x: -x[2])[:8]
    c["matcher"] = matcher
    c["pairs"] = {"sift": len(keep), "sift_dropped": dropped_sift, "dense": len(dense), "overlapping": len(_overlapping(P))}
    json.dump(c, open(out, "w"), indent=1)
    print(f"pairs: {len(keep)} {matcher} ({dropped_sift} inconsistent dropped) + {len(dense)} dense of {len(_overlapping(P))} overlapping; focal {c['focal_px']:.0f} px, "
          f"{c['deg_per_step_pan']:.4f}/{c['deg_per_step_tilt']:.4f} deg/step, level tilt {c['level_tilt']:.0f}; "
          f"median pair error {c['median_px']:.1f} px; worst {c['worst_pairs'][:3]}")
    return c


if __name__ == "__main__":
    calibrate(*sys.argv[1:3])
