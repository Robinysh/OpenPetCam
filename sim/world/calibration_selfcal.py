"""Natural-scene optical calibration with free orientations and a valid lens map.

This alternate fitter uses saved training-sweep matches only. It jointly fits
f,k1,k2 and gauge-fixed frame rotations with symmetric pixel residuals. Positive
Bernstein coefficients of the radial derivative guarantee a monotonic map out
to normalized radius0.9. Focal length is parameterized to retain the full sensor
diagonal inside that invertible domain. No validation images are loaded here.

    python3 -m sim.world.calibration_selfcal --sweep eval/twin/day \
      --matches eval/world/raw/matches_lightglue.npz \
      --seed-profile eval/world_remap1/calib.json \
      --out eval/calibration_20261106/candidate_selfcal.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration_validation import (W, H, _digest, _flatten, _initialize_rotations,
    _right_jacobian, _skew, load_matches, nominal_rotations, undistort_points)

RADIUS = .9
DIAGONAL = float(np.hypot(W / 2, H / 2))


def optics_parameters(profile):
    k1, k2, f = (profile[k] for k in ("k1", "k2", "focal_px"))
    b1 = 1 + 1.5 * k1 * RADIUS**2
    b2 = 1 + 3 * k1 * RADIUS**2 + 5 * k2 * RADIUS**4
    rd = RADIUS * (1 + k1 * RADIUS**2 + k2 * RADIUS**4)
    return np.array([np.log(f * rd / DIAGONAL), b1, b2])


def optical_profile(parameters, seed):
    h, b1, b2 = parameters
    k1 = (b1 - 1) / (1.5 * RADIUS**2)
    k2 = (b2 - 2 * b1 + 1) / (5 * RADIUS**4)
    rd = RADIUS * (1 + k1 * RADIUS**2 + k2 * RADIUS**4)
    f = DIAGONAL * np.exp(h) / rd
    return dict(seed, focal_px=float(f), k1=float(k1), k2=float(k2))


def project_with_jac(rays, profile):
    """Full projection; extend past the calibrated domain monotonically.

    The linear continuation is optimizer-only. Measured image rays must all be
    invertible and final predictions must be in front. Outlying matches whose
    predictions leave radius0.9 remain penalized and are counted in diagnostics.
    """
    z = np.maximum(rays[:, 2], .05)
    xy = rays[:, :2] / z[:, None]
    r = np.linalg.norm(xy, axis=1)
    r2 = r * r
    k1, k2, f = (profile[k] for k in ("k1", "k2", "focal_px"))
    rd = r * (1 + k1 * r2 + k2 * r2**2)
    deriv = 1 + 3 * k1 * r2 + 5 * k2 * r2**2
    outer = r > RADIUS
    rdmax = RADIUS * (1 + k1 * RADIUS**2 + k2 * RADIUS**4)
    qmax = 1 + 3 * k1 * RADIUS**2 + 5 * k2 * RADIUS**4
    rd[outer] = rdmax + (r[outer] - RADIUS) * qmax
    deriv[outer] = qmax
    scale = np.divide(rd, r, out=np.ones_like(r), where=r > 1e-12)
    alpha = np.divide(deriv * r - rd, r**3, out=np.full_like(r, 2 * k1), where=r > 1e-8)
    uv = xy * (f * scale)[:, None] * [1, -1] + [W / 2, H / 2]
    plane = np.zeros((len(r), 2, 2))
    plane[:, 0, 0] = scale + alpha * xy[:, 0]**2
    plane[:, 0, 1] = alpha * xy[:, 0] * xy[:, 1]
    plane[:, 1, 0] = -plane[:, 0, 1]
    plane[:, 1, 1] = -scale - alpha * xy[:, 1]**2
    plane *= f
    jac = np.concatenate([plane / z[:, None, None],
                          -np.einsum("nij,nj->ni", plane, xy)[:, :, None] / z[:, None, None]], axis=2)
    jac[rays[:, 2] <= .05, :, 2] = 0.
    valid = (r <= RADIUS) & (rays[:, 2] > .05) & np.isfinite(uv).all(axis=1)
    return uv, jac, valid


def fit_motor_model(poses, pairs, rotations, profile):
    """Fit the motor nuisance model after optics, from relative rotations only.

    Component-specific world gauges therefore cannot bias the motor fit. This
    simple model is still an average: it does not identify directional backlash.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    ii = np.array([i for i, _, _, _ in pairs])
    jj = np.array([j for _, j, _, _ in pairs])
    measured = np.swapaxes(rotations[ii], 1, 2) @ rotations[jj]
    names = ("deg_per_step_pan", "deg_per_step_tilt", "level_tilt", "roll_deg")
    def residual(th):
        p = dict(profile, **dict(zip(names, th)))
        rr = nominal_rotations(poses, p)
        predicted = np.swapaxes(rr[ii], 1, 2) @ rr[jj]
        return np.degrees(Rotation.from_matrix(np.swapaxes(predicted, 1, 2) @ measured).as_rotvec()).ravel()
    result = least_squares(residual, [profile[k] for k in names],
                           bounds=([.05, .05, 400., -15.], [.13, .13, 1000., 15.]),
                           x_scale=[.01, .01, 100., 2.], loss="soft_l1", f_scale=.5, max_nfev=300,
                           ftol=1e-9, xtol=1e-9, gtol=1e-7)
    out = dict(profile, **dict(zip(names, map(float, result.x))))
    err = np.linalg.norm(result.fun.reshape(-1, 3), axis=1)
    return out, {"success": bool(result.success), "nfev": result.nfev,
                 "relative_rotation_median_deg": float(np.median(err)),
                 "relative_rotation_p95_deg": float(np.percentile(err, 95)),
                 "limitations": "average motor model; directional backlash and pivot translation not identified"}


def fit(sweep, matches, seed_profile, out, max_nfev=180):
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix
    from scipy.spatial.transform import Rotation
    poses = json.loads((Path(sweep) / "poses.json").read_text())["poses"]
    seed = json.loads(Path(seed_profile).read_text())
    pairs = load_matches(matches, poses)
    ii, jj, a, b = _flatten(pairs)
    n, m = len(poses), len(a)
    ra, va = undistort_points(a, seed)
    rb, vb = undistort_points(b, seed)
    if not (va & vb).all():
        raise ValueError("seed does not cover training observations")
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j, _, _ in pairs:
        parent[find(i)] = find(j)
    anchors = {min(i for i in range(n) if find(i) == c) for c in {find(i) for i in range(n)}}
    free = np.array([i for i in range(n) if i not in anchors])
    col = np.full(n, -1, int); col[free] = np.arange(len(free))
    nominal = _initialize_rotations(nominal_rotations(poses, seed), pairs, ra, rb, seed["focal_px"], anchors)
    lower = np.r_[np.log(1.005), .2, .02, np.full(3 * len(free), -.3)]
    upper = np.r_[np.log(1.5), 1.5, 1.5, np.full(3 * len(free), .3)]
    initial = np.r_[optics_parameters(seed), np.zeros(3 * len(free))]
    initial = np.clip(initial, lower + 1e-8, upper - 1e-8)
    cache = {}
    def state(th):
        if "th" in cache and np.array_equal(cache["th"], th):
            return cache
        p = optical_profile(th[:3], seed)
        ra, va = undistort_points(a, p); rb, vb = undistort_points(b, p)
        rr = nominal.copy()
        rr[free] = nominal[free] @ Rotation.from_rotvec(th[3:].reshape(-1, 3)).as_matrix()
        wa = np.einsum("nij,nj->ni", rr[ii], ra)
        wb = np.einsum("nij,nj->ni", rr[jj], rb)
        ab = np.einsum("nji,nj->ni", rr[jj], wa)
        ba = np.einsum("nji,nj->ni", rr[ii], wb)
        uvb, jab, vab = project_with_jac(ab, p)
        uva, jba, vba = project_with_jac(ba, p)
        residual = np.r_[(uvb - b).ravel(), (uva - a).ravel()] / np.sqrt(2.)
        cache.clear(); cache.update(th=th.copy(), p=p, ra=ra, rb=rb, rr=rr, ab=ab, ba=ba,
                                    jab=jab, jba=jba, residual=residual, valid=va & vb & vab & vba,
                                    inverse_valid=va & vb, front_valid=(ab[:, 2] > .05) & (ba[:, 2] > .05))
        return cache
    def residual(th):
        # SciPy may rescale residual arrays in-place for a robust loss.
        return state(th)["residual"].copy()
    def jacobian(th):
        # The three optical parameters use centred finite differences through
        # the independent inverse radial map; rotation blocks are analytic.
        data, rows, columns = [], [], []
        for k in range(3):
            delta = min(1e-6, (upper[k] - lower[k]) * 1e-5)
            lo, hi = th.copy(), th.copy()
            lo[k] = max(lower[k], th[k] - delta); hi[k] = min(upper[k], th[k] + delta)
            derivative = (residual(hi) - residual(lo)) / (hi[k] - lo[k])
            data.append(derivative); rows.append(np.arange(4 * m)); columns.append(np.full(4 * m, k))
        s = state(th)
        rr, ra, rb = s["rr"], s["ra"], s["rb"]
        jr = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        jr[free] = _right_jacobian(th[3:].reshape(-1, 3))
        # Forward: framei is the source, framej the inverse transform.
        blocks = [(ii, 0, s["jab"] @ (np.swapaxes(rr[jj], 1, 2) @ (-rr[ii] @ _skew(ra) @ jr[ii]))),
                  (jj, 0, s["jab"] @ (_skew(s["ab"]) @ jr[jj])),
                  (jj, 2 * m, s["jba"] @ (np.swapaxes(rr[ii], 1, 2) @ (-rr[jj] @ _skew(rb) @ jr[jj]))),
                  (ii, 2 * m, s["jba"] @ (_skew(s["ba"]) @ jr[ii]))]
        for frames, base, block in blocks:
            use = col[frames] >= 0
            indexes = np.flatnonzero(use)
            rows.append(base + np.repeat(indexes * 2, 6) + np.tile(np.repeat(np.arange(2), 3), len(indexes)))
            columns.append(3 + np.repeat(col[frames[use]] * 3, 6) + np.tile(np.tile(np.arange(3), 2), len(indexes)))
            data.append(block[use].ravel() / np.sqrt(2.))
        return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
                          shape=(4 * m, 3 + 3 * len(free))).tocsr()
    result = least_squares(residual, initial, jac=jacobian, bounds=(lower, upper), loss="soft_l1", f_scale=3.,
                           x_scale="jac", max_nfev=max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-6,
                           tr_options={"atol": 1e-10, "btol": 1e-10, "maxiter": 1500})
    s = state(result.x)
    p = s["p"]
    forward = s["residual"][:2 * m].reshape(m, 2)
    backward = s["residual"][2 * m:].reshape(m, 2)
    errors = np.sqrt((forward * forward).sum(axis=1) + (backward * backward).sum(axis=1))
    per_pair, at = [], 0
    for _, _, aa, _ in pairs:
        per_pair.append(float(np.median(errors[at:at + len(aa)])))
        at += len(aa)
    diagnostics = {"success": bool(result.success), "message": result.message, "nfev": result.nfev,
                   "cost": float(result.cost), "optimality": float(result.optimality),
                   "parameters": result.x[:3].tolist(), "parameters_on_bound": result.active_mask[:3].tolist(),
                   "training_pairs": len(pairs), "training_matches": m, "invalid_training_projections": int((~s["valid"]).sum()),
                   "noninvertible_measured_matches": int((~s["inverse_valid"]).sum()),
                   "nonfront_predictions": int((~s["front_valid"]).sum()),
                   "training_median_symmetric_px": float(np.median(errors)),
                   "training_p90_symmetric_px": float(np.percentile(errors, 90)),
                   "training_p95_symmetric_px": float(np.percentile(errors, 95)),
                   "out_of_domain_policy": "optimizer-only monotonic linear continuation beyond radius0.9; every observation remains penalized",
                   "training_sweep": str(sweep), "poses_sha256": _digest(Path(sweep) / "poses.json"),
                   "matches_sha256": _digest(matches), "seed_profile_sha256": _digest(seed_profile),
                   "fitter_sha256": _digest(__file__), "validation_images_used": False,
                   "model": "symmetric pixel fit; monotonic two-radial optics; independent free frame rotations"}
    print(json.dumps({k: p[k] for k in ("focal_px", "k1", "k2")}), json.dumps(diagnostics), flush=True)
    # Keep a diagnostic file even for rejected fits; no incomplete optical
    # profile gets mistaken for a usable candidate by the validation CLI.
    dest = Path(out); dest.parent.mkdir(parents=True, exist_ok=True)
    Path(str(dest) + ".fit.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    if not result.success or not s["inverse_valid"].all() or not s["front_valid"].all() or not np.isfinite(s["residual"]).all():
        raise RuntimeError("self-calibration did not converge to an invertible, finite, front-facing profile")
    p, motor_diagnostics = fit_motor_model(poses, pairs, s["rr"], p)
    diagnostics["motor_fit"] = motor_diagnostics
    Path(str(dest) + ".fit.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    for k in ("offsets_deg", "frames", "median_px", "worst_pairs", "pairs", "matcher"):
        p.pop(k, None)
    # Express data-fitted rotations in the existing correction convention.
    # Motor constants remain the seed nuisance model; offsets supply the exact
    # training orientations rather than mixing optical estimation with motors.
    base = nominal_rotations(poses, p)
    offsets = []
    for r0, r1 in zip(base, s["rr"]):
        # Camera.R uses Ry(yawoffset) Rx(pitchoffset) Rz(rolloffset).
        angles = Rotation.from_matrix(r0.T @ r1).as_euler("YXZ", degrees=True)
        offsets.append([float(angles[1]), float(angles[0]), float(angles[2])])
    p["offsets_deg"] = offsets; p["frames"] = [v["file"] for v in poses]
    p["median_px"] = float(np.median(per_pair))
    p["pairs"] = {"lightglue": len(pairs), "matches": m}
    p["matcher"] = "ALIKED+LightGlue; symmetric monotonic self-calibration"
    p["selfcal"] = diagnostics
    dest.write_text(json.dumps(p, indent=2) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--seed-profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-nfev", type=int, default=180)
    a = ap.parse_args()
    fit(a.sweep, a.matches, a.seed_profile, a.out, a.max_nfev)


if __name__ == "__main__":
    main()
