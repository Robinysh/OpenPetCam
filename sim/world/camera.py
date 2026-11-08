"""The camera as a measuring instrument: a rotation-only pan/tilt camera calibrated from its own sweep.

World frame: the lens at the origin, y up, z forward at pan 1850 (yaw 0), x right. Left-handed (like a camera with y flipped up): renderers that are right-handed (three.js, Blender) must mirror one axis. View right = lower pan, picture up
= higher tilt. calibrate() fits focal length, two radial distortion terms, degrees per motor step (pan, tilt), the
level tilt, roll and a per-frame pose correction (backlash, end stops) from SIFT matches plus dense edge matches between
overlapping frames (white walls give SIFT almost nothing).
The saved profile is an empirical fit, not a factory calibration. Its in-sample pair error does not establish
accuracy on another sweep; use calibration_validation.py with a separate capture for that check.

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
        # Stronger barrel profiles converge slowly at the image corners.
        # Keep this iteration count equal to server/roomcam.py and the fitter.
        for _ in range(30):
            r2 = xu * xu + yu * yu
            k = 1 + self.k1 * r2 + self.k2 * r2 * r2
            xu, yu = xd / k, yd / k
        v = np.stack([xu, yu, np.ones_like(xu)], -1) @ self.R(pan, tilt, off).T
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def project_unbounded(self, pan, tilt, d, w=None, h=None, off=(0, 0, 0)):
        """Project directions without the renderer's image/radius clipping.

        The last result marks finite forward geometry. Calibration must score an
        out-of-frame prediction, rather than let it disappear behind a crop mask.
        """
        w, h = w or self.W, h or self.H
        c = np.asarray(d) @ self.R(pan, tilt, off)
        front = c[..., 2] > 0.15
        z = np.where(front, c[..., 2], 1)
        x, y = c[..., 0] / z, c[..., 1] / z
        r2 = x * x + y * y
        k = 1 + self.k1 * r2 + self.k2 * r2 * r2
        u = (x * k * self.f + self.W / 2) * w / self.W
        v = (self.H / 2 - y * k * self.f) * w / self.W
        return u, v, front & np.isfinite(u) & np.isfinite(v)

    def project(self, pan, tilt, d, w=None, h=None, off=(0, 0, 0)):
        """World directions -> pixels and the renderer's conservative image mask."""
        w, h = w or self.W, h or self.H
        u, v, front = self.project_unbounded(pan, tilt, d, w, h, off)
        c = np.asarray(d) @ self.R(pan, tilt, off)
        z = np.where(c[..., 2] > 0.15, c[..., 2], 1)
        r2 = np.sum((c[..., :2] / z[..., None]) ** 2, axis=-1)
        return u, v, front & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) & (r2 < 0.45)


def calibration_residuals(cam, poses, pairs, offsets, scale=0.5):
    """Symmetric pixel errors for every supplied correspondence, including invalid ones.

    Pair/match selection is fixed before evaluation. A prediction outside the
    image still has its full pixel error; back-facing or nonfinite projections
    receive a finite positive penalty instead of the old zero residual.
    """
    out = []
    for i, j, a, b in pairs:
        for source, target, xy, want in ((i, j, a, b), (j, i, b, a)):
            p, q = poses[source], poses[target]
            d = cam.rays(p["pan"], p["tilt"], xy[:, 0], xy[:, 1], off=offsets[source])
            u, v, front = cam.project_unbounded(q["pan"], q["tilt"], d, off=offsets[target])
            err = np.stack([u, v], axis=-1) - want
            z = (d @ cam.R(q["pan"], q["tilt"], offsets[target]))[:, 2]
            penalty = cam.W * (1 + np.maximum(0, 0.15 - np.nan_to_num(z, nan=-1.0)))
            err = np.where(front[:, None], err, penalty[:, None])
            out.append(np.nan_to_num(err, nan=cam.W, posinf=10 * cam.W, neginf=-10 * cam.W).ravel() * scale)
    return np.concatenate(out) if out else np.empty(0)


class _CalibrationObservations:
    """Vectorized equivalent of calibration_residuals for repeated solver calls."""
    def __init__(self, poses, pairs):
        self.poses = poses
        source, target, points, wanted = [], [], [], []
        for i, j, a, b in pairs:
            for p, q, xy, want in ((i, j, a, b), (j, i, b, a)):
                source.extend([p] * len(xy)); target.extend([q] * len(xy))
                points.append(xy); wanted.append(want)
        self.source, self.target = np.asarray(source), np.asarray(target)
        self.points, self.wanted = np.concatenate(points), np.concatenate(wanted)

    def residuals(self, cam, offsets, scale=0.5):
        r = np.array([cam.R(p["pan"], p["tilt"], off) for p, off in zip(self.poses, offsets)])
        xd = (self.points - [cam.W / 2, cam.H / 2]) / [cam.f, -cam.f]
        xy = xd.copy()
        for _ in range(30):
            r2 = np.sum(xy * xy, axis=1)
            xy = xd / (1 + cam.k1 * r2 + cam.k2 * r2 * r2)[:, None]
        rays = np.column_stack([xy, np.ones(len(xy))])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        world = np.einsum("nij,nj->ni", r[self.source], rays)
        local = np.einsum("nji,nj->ni", r[self.target], world)
        z = local[:, 2]
        xy = local[:, :2] / np.where(z > 0.15, z, 1)[:, None]
        r2 = np.sum(xy * xy, axis=1)
        uv = xy * (1 + cam.k1 * r2 + cam.k2 * r2 * r2)[:, None] * [cam.f, -cam.f] + [cam.W / 2, cam.H / 2]
        front = (z > 0.15) & np.isfinite(uv).all(axis=1)
        penalty = cam.W * (1 + np.maximum(0, 0.15 - np.nan_to_num(z, nan=-1.0)))
        err = np.where(front[:, None], uv - self.wanted, penalty[:, None])
        return np.nan_to_num(err, nan=cam.W, posinf=10 * cam.W, neginf=-10 * cam.W).ravel() * scale


_PARAMS = ("focal_px", "k1", "k2", "deg_per_step_pan", "deg_per_step_tilt", "level_tilt", "roll_deg")
_INITIAL = (1747, -0.24, -0.11, 0.0843, 0.0803, 728, 0.3)


def _radial_constraints(cam):
    # The full image corner must have a unique inverse on this interval. Bound
    # the derivative as well as the coefficients: k bounds alone are insufficient.
    # Its derivative is quadratic in s=r²; checking its vertex and endpoints
    # establishes the minimum exactly, including between the sample positions.
    vertex = np.clip(-3 * cam.k1 / (10 * cam.k2), 0, 0.81) if cam.k2 > 0 else 0
    s = np.r_[np.linspace(0, 0.81, 24), vertex]
    derivative = 1 + 3 * cam.k1 * s + 5 * cam.k2 * s * s
    corner = math.hypot(cam.W / 2, cam.H / 2) / cam.f
    reach = 0.9 * (1 + cam.k1 * 0.81 + cam.k2 * 0.81 ** 2)
    return np.r_[np.maximum(0.02 - derivative, 0), max(corner + 0.005 - reach, 0)] * cam.W


def fit_matches(poses, pairs, initial=None, prior_deg=3.0, max_nfev=300):
    """Fit the supported radial model and motor orientations to fixed matches.

    This has no image I/O or candidate-dependent match rejection, so an offline
    experiment can train on one capture and reserve another capture for scoring.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    if not pairs or prior_deg <= 0:
        raise ValueError("Calibration requires matched pairs and a positive orientation prior")
    n = len(poses)
    initial = initial or dict(zip(_PARAMS, _INITIAL))
    frame_offsets = dict(zip(initial.get("frames", []), initial.get("offsets_deg", [])))
    off0 = np.array([frame_offsets.get(p["file"], (0, 0, 0)) for p in poses], float)
    th0 = np.r_[[initial[k] for k in _PARAMS], off0.ravel()]
    observations = _CalibrationObservations(poses, pairs)

    def unpack(th):
        return dict(zip(_PARAMS, th[:7])), th[7:].reshape(n, 3)

    def res(th):
        c, off = unpack(th)
        cam = Camera(c)
        return np.r_[observations.residuals(cam, off), off.ravel() / prior_deg, 1000 * _radial_constraints(cam)]

    def loss(z):
        # Robustify image/pose observations, never physical-validity constraints.
        root = np.sqrt(1 + z)
        rho = np.array([2 * (root - 1), 1 / root, -0.5 / root ** 3])
        rho[:, -26:] = np.array([z[-26:], np.ones(26), np.zeros(26)])
        return rho

    rows = 4 * sum(len(a) for _i, _j, a, _b in pairs)
    sparsity = lil_matrix((rows + 3 * n + 26, 7 + 3 * n), dtype=int)
    start = 0
    for i, j, a, _b in pairs:
        end = start + 4 * len(a)
        sparsity[start:end, :7] = 1
        sparsity[start:end, 7 + 3 * i:10 + 3 * i] = 1
        sparsity[start:end, 7 + 3 * j:10 + 3 * j] = 1
        start = end
    for k in range(3 * n):
        sparsity[rows + k, 7 + k] = 1
    sparsity[rows + 3 * n:, :3] = 1
    lo = np.r_[[960, -0.7, -0.7, 0.04, 0.04, 0, -15], np.full(3 * n, -15)]
    hi = np.r_[[7680, 0.7, 0.7, 0.14, 0.14, 1100, 15], np.full(3 * n, 15)]
    xs = np.r_[[100, 0.05, 0.05, 0.002, 0.002, 30, 1], np.ones(3 * n)]
    result = least_squares(res, th0, bounds=(lo, hi), loss=loss, f_scale=3.0,
                           x_scale=xs, jac_sparsity=sparsity.tocsr(), max_nfev=max_nfev)
    c, off = unpack(result.x)
    c = {k: float(v) for k, v in c.items()}
    if np.max(_radial_constraints(Camera(c))) > 0.1:
        raise ValueError(f"Fitted radial profile is nonmonotonic or does not cover the full image: {c}")
    return c, off, result


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

    def solve(pp, initial):
        return fit_matches(P, pp, initial, prior_deg)

    def with_offsets(c, off):
        return dict(c, frames=[p["file"] for p in P], offsets_deg=off.tolist())

    c, off, result = solve(pairs, None)
    def pair_errors(c, off, pp):
        return [float(np.median(np.linalg.norm(calibration_residuals(Camera(c), P, [pair], off, scale=1).reshape(-1, 2), axis=1)))
                for pair in pp]

    # SIFT pairs that the first fit cannot explain are wrong matches (repeating doors, tiles): drop and refit
    keep = [p_ for p_, e in zip(pairs, pair_errors(c, off, pairs)) if e < 12]
    dropped_sift = len(pairs) - len(keep)
    c, off, result = solve(keep, with_offsets(c, off))
    dense = _dense_pairs(Camera(c), P, imgs, S, off)
    dense = [p_ for p_, e in zip(dense, pair_errors(c, off, dense)) if e < 6]     # dense matches only refine
    allp = keep + dense
    c, off, result = solve(allp, with_offsets(c, off))
    per = pair_errors(c, off, allp)
    c = {k: float(v) for k, v in c.items()}
    c["offsets_deg"] = off.tolist()
    c["frames"] = [p["file"] for p in P]
    c["median_px"] = float(np.median(per))
    c["worst_pairs"] = sorted(([P[i]["file"], P[j]["file"], round(e, 1)] for (i, j, _a, _b), e in zip(allp, per)),
                              key=lambda x: -x[2])[:8]
    c["matcher"] = matcher
    c["pairs"] = {"sift": len(keep), "sift_dropped": dropped_sift, "dense": len(dense), "overlapping": len(_overlapping(P))}
    c["fit"] = {"objective": "symmetric pixel error, no crop-mask rejection", "success": bool(result.success),
                "nfev": int(result.nfev), "error_scope": "in-sample median of pair medians"}
    json.dump(c, open(out, "w"), indent=1)
    print(f"pairs: {len(keep)} {matcher} ({dropped_sift} inconsistent dropped) + {len(dense)} dense of {len(_overlapping(P))} overlapping; focal {c['focal_px']:.0f} px, "
          f"{c['deg_per_step_pan']:.4f}/{c['deg_per_step_tilt']:.4f} deg/step, level tilt {c['level_tilt']:.0f}; "
          f"median pair error {c['median_px']:.1f} px; worst {c['worst_pairs'][:3]}")
    return c


if __name__ == "__main__":
    calibrate(*sys.argv[1:3])
