"""Independent optical-profile validation on an existing, separate camera sweep.

The candidate's intrinsics are frozen. Frame rotations are nuisance parameters,
estimated on a connected set of image pairs. Scores use other image pairs AND
other feature coordinates, with a split fixed before looking at any candidate.
The projection implementation deliberately does not import camera.Camera.

    python3 -m sim.world.calibration_validation --sweep eval/twin/dense \
      --matches eval/world_dense/raw/matches_lightglue.npz \
      --profile old=eval/world_remap1/calib.json --profile new=path/to/new.json \
      --training-sweep eval/twin/day --out eval/calibration_20261106/validation.json

These are cross-image consistency checks, not known-target calibration truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

W, H = 1920, 1080


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sweep_hashes(sweep):
    poses = json.loads((Path(sweep) / "poses.json").read_text())["poses"]
    return {_digest(Path(sweep) / p["file"]) for p in poses}


def capture_overlap(training_sweep, validation_sweep):
    a, b = sweep_hashes(training_sweep), sweep_hashes(validation_sweep)
    return {"training_unique_frames": len(a), "validation_unique_frames": len(b),
            "shared_frame_sha256_count": len(a & b), "independent_captures": not bool(a & b)}


def load_matches(path, poses, seed=20261106):
    """One candidate-independent RANSAC mask, in the raw distorted image plane."""
    import cv2
    pairs = []
    with np.load(path) as z:
        for key in sorted(z.files, key=lambda k: tuple(map(int, k.split("_")))):
            i, j = map(int, key.split("_"))
            if i >= len(poses) or j >= len(poses):
                raise ValueError("match archive references a missing frame")
            m = np.asarray(z[key], float)
            valid = np.isfinite(m).all(axis=1)
            valid &= ((m[:, [0, 2]] >= 0) & (m[:, [0, 2]] < W)).all(axis=1)
            valid &= ((m[:, [1, 3]] >= 0) & (m[:, [1, 3]] < H)).all(axis=1)
            # Exclude timestamp features, rather than giving their stationary
            # image coordinates the interpretation of stationary room points.
            valid &= ~(((m[:, 0] < .36 * W) & (m[:, 1] < .05 * H)) |
                       ((m[:, 2] < .36 * W) & (m[:, 3] < .05 * H)))
            m = m[valid]
            if len(m) < 12:
                continue
            cv2.setRNGSeed(seed % (2**31 - 1))
            _, inlier = cv2.findHomography(m[:, :2] * .5, m[:, 2:] * .5, cv2.RANSAC, 4.)
            if inlier is not None and inlier.sum() >= 12:
                m = m[inlier.ravel().astype(bool)]
                pairs.append((i, j, m[:, :2], m[:, 2:]))
    return pairs


def _feature_fit_mask(frame, points, seed):
    # Identical feature coordinates in the same image get the same assignment
    # across every edge of the match graph. Quantization handles stored floats.
    q = np.rint(np.asarray(points) * 100).astype(np.int64)
    return np.array([int.from_bytes(hashlib.blake2b(
        f"{seed}:{frame}:{x}:{y}".encode(), digest_size=8).digest(), "little") % 2 == 0
        for x, y in q], bool)


def validation_split(pairs, n_frames, seed=20261106, score_share=.3):
    """A maximum-weight spanning forest keeps the pose-fit graph connected.

    Half the feature coordinates are reserved for scoring before choosing edges.
    A match is usable only when both endpoints have the required assignment.
    """
    prepared = []
    for i, j, a, b in pairs:
        ma, mb = _feature_fit_mask(i, a, seed), _feature_fit_mask(j, b, seed)
        train = ma & mb
        score = ~ma & ~mb
        if train.sum() >= 4:
            prepared.append((i, j, a, b, train, score))
    parent = list(range(n_frames))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = set()
    for k in sorted(range(len(prepared)), key=lambda k: (-prepared[k][4].sum(), prepared[k][:2])):
        i, j = prepared[k][:2]
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
            tree.add(k)
    train, score = [], []
    for k, (i, j, a, b, mt, ms) in enumerate(prepared):
        h = int.from_bytes(hashlib.blake2b(f"{seed}:pair:{i}:{j}".encode(), digest_size=8).digest(), "little")
        held_out = k not in tree and h / 2**64 < score_share and ms.sum() >= 4
        if held_out:
            score.append((i, j, a[ms], b[ms]))
        else:
            train.append((i, j, a[mt], b[mt]))
    if not train or not score:
        raise ValueError("not enough independent pair/feature observations for a validation split")
    return train, score


def _radial_turning_radius(profile):
    k1, k2 = float(profile["k1"]), float(profile["k2"])
    roots = np.roots([5 * k2, 3 * k1, 1]) if k2 else np.roots([3 * k1, 1]) if k1 else []
    positive = [float(np.sqrt(r.real)) for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    return min(positive) if positive else np.inf


def undistort_points(points, profile):
    """Invert the radial map, retaining failures so they cannot improve a score."""
    f = float(profile["focal_px"])
    k1, k2 = float(profile["k1"]), float(profile["k2"])
    q = (np.asarray(points, float) - [W / 2, H / 2]) / f
    rd = np.linalg.norm(q, axis=1)
    # Bracket the first monotonic branch of rd=r*(1+k1*r²+k2*r⁴).
    turning = _radial_turning_radius(profile)
    hi = turning * (1 - 1e-10) if np.isfinite(turning) else max(4., float(rd.max(initial=0)) * 2 + 1)
    max_rd = hi * (1 + k1 * hi**2 + k2 * hi**4)
    good = np.isfinite(q).all(axis=1) & (rd <= max_rd) & (f > 0)
    lo_a, hi_a = np.zeros_like(rd), np.full_like(rd, hi)
    for _ in range(48):
        mid = (lo_a + hi_a) * .5
        lower = mid * (1 + k1 * mid**2 + k2 * mid**4) < rd
        lo_a = np.where(lower, mid, lo_a)
        hi_a = np.where(lower, hi_a, mid)
    ru = (lo_a + hi_a) * .5
    xy = q * np.divide(ru, rd, out=np.ones_like(rd), where=rd > 0)[:, None]
    rays = np.column_stack([xy[:, 0], -xy[:, 1], np.ones(len(q))])
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    return rays, good


def project_bearings(rays, profile):
    rays = np.asarray(rays)
    front = rays[:, 2] > 1e-6
    z = np.where(front, rays[:, 2], 1e-6)
    xy = rays[:, :2] / z[:, None]
    r2 = (xy * xy).sum(axis=1)
    scale = 1 + profile["k1"] * r2 + profile["k2"] * r2**2
    uv = xy * (profile["focal_px"] * scale)[:, None]
    uv = uv * [1, -1] + [W / 2, H / 2]
    # Past its first turning point the polynomial folds back toward the image.
    # Keep such predictions in the scored denominator with the invalid penalty.
    monotonic = np.sqrt(r2) < _radial_turning_radius(profile) * (1 - 1e-10)
    return uv, front & monotonic & np.isfinite(uv).all(axis=1)


def nominal_rotations(poses, profile):
    """Reimplement motor geometry; do not inherit the calibrated renderer mask."""
    from scipy.spatial.transform import Rotation
    yaw = np.radians([(1850 - p["pan"]) * profile["deg_per_step_pan"] for p in poses])
    pitch = np.radians([-(p["tilt"] - profile["level_tilt"]) * profile["deg_per_step_tilt"] for p in poses])
    ry = Rotation.from_rotvec(np.column_stack([np.zeros(len(poses)), yaw, np.zeros(len(poses))])).as_matrix()
    rx = Rotation.from_rotvec(np.column_stack([pitch, np.zeros(len(poses)), np.zeros(len(poses))])).as_matrix()
    rz = Rotation.from_rotvec([0, 0, np.radians(profile["roll_deg"])]).as_matrix()
    return ry @ rx @ rz


def _skew(v):
    out = np.zeros((len(v), 3, 3))
    out[:, 0, 1], out[:, 0, 2] = -v[:, 2], v[:, 1]
    out[:, 1, 0], out[:, 1, 2] = v[:, 2], -v[:, 0]
    out[:, 2, 0], out[:, 2, 1] = -v[:, 1], v[:, 0]
    return out


def _right_jacobian(rotvec):
    """Derivative of SO(3)'s exponential in its local tangent coordinates."""
    t2 = (rotvec * rotvec).sum(axis=1)
    t = np.sqrt(t2)
    a = np.full_like(t, .5)
    b = np.full_like(t, 1 / 6)
    big = t > 1e-5
    a[big] = (1 - np.cos(t[big])) / t2[big]
    b[big] = (t[big] - np.sin(t[big])) / (t2[big] * t[big])
    small = ~big
    a[small] -= t2[small] / 24
    b[small] -= t2[small] / 120
    sk = _skew(rotvec)
    return np.eye(3)[None] - a[:, None, None] * sk + b[:, None, None] * (sk @ sk)


def _initialize_rotations(nominal, pairs, ra, rb, focal, anchors):
    """Data-only relative rotations avoid a slow/motor-dependent starting pose."""
    links = []
    offset = 0
    for i, j, a, _ in pairs:
        aa, bb = ra[offset:offset + len(a)], rb[offset:offset + len(a)]
        offset += len(a)
        weight = np.ones(len(a))
        for _ in range(6):
            u, _, vt = np.linalg.svd((aa * weight[:, None]).T @ bb)
            q = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
            err = np.linalg.norm(aa - bb @ q.T, axis=1) * focal
            weight = np.minimum(1., 3. / np.maximum(err, 1e-8))
        links.append((len(a), i, j, q))
    parent = list(range(len(nominal)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    adjacency = [[] for _ in nominal]
    for _, i, j, q in sorted(links, key=lambda x: (-x[0], x[1], x[2])):
        if find(i) != find(j):
            parent[find(i)] = find(j)
            adjacency[i].append((j, q)); adjacency[j].append((i, q.T))
    out = nominal.copy()
    visited = set(anchors)
    todo = list(anchors)
    while todo:
        i = todo.pop()
        for j, q in adjacency[i]:
            if j not in visited:
                out[j] = out[i] @ q
                visited.add(j); todo.append(j)
    return out


def fit_rotations(poses, pairs, profile, max_nfev=120):
    """Estimate orientations only, using robust spherical bearing agreement."""
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix
    from scipy.spatial.transform import Rotation
    n = len(poses)
    nominal = nominal_rotations(poses, profile)
    ii, jj, a, b = _flatten(pairs)
    ra, va = undistort_points(a, profile)
    rb, vb = undistort_points(b, profile)
    if not (va & vb).all():
        raise ValueError("profile cannot invert every orientation-fit observation")
    # One fixed orientation per connected component removes the global-rotation
    # gauge. Its particular value cannot affect cross-image reprojection error.
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j, _, _ in pairs:
        parent[find(i)] = find(j)
    anchors = {min(i for i in range(n) if find(i) == c) for c in {find(i) for i in range(n)}}
    nominal = _initialize_rotations(nominal, pairs, ra, rb, profile["focal_px"], anchors)
    free = np.array([i for i in range(n) if i not in anchors])
    col = np.full(n, -1, int)
    col[free] = np.arange(len(free))
    ska, skb = _skew(ra), _skew(rb)
    def rotations(th):
        out = nominal.copy()
        out[free] = nominal[free] @ Rotation.from_rotvec(th.reshape(-1, 3)).as_matrix()
        return out
    def residual(th):
        rr = rotations(th)
        wa = np.einsum("nij,nj->ni", rr[ii], ra)
        wb = np.einsum("nij,nj->ni", rr[jj], rb)
        return ((wa - wb) * profile["focal_px"]).ravel()
    def jacobian(th):
        rr = rotations(th)
        jr = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        jr[free] = _right_jacobian(th.reshape(-1, 3))
        rows, columns, data = [], [], []
        for frames, sk, sign in [(ii, ska, -1), (jj, skb, 1)]:
            use = col[frames] >= 0
            indexes = np.flatnonzero(use)
            block = sign * profile["focal_px"] * (rr[frames[use]] @ sk[use] @ jr[frames[use]])
            rows.append(np.repeat(indexes * 3, 9) + np.tile(np.repeat(np.arange(3), 3), len(indexes)))
            columns.append(np.repeat(col[frames[use]] * 3, 9) + np.tile(np.tile(np.arange(3), 3), len(indexes)))
            data.append(block.ravel())
        return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
                          shape=(3 * len(a), 3 * len(free))).tocsr()
    result = least_squares(residual, np.zeros(3 * len(free)), jac=jacobian,
                           loss="soft_l1", f_scale=3., x_scale="jac", max_nfev=max_nfev,
                           tr_options={"atol": 1e-10, "btol": 1e-10, "maxiter": 1500},
                           ftol=1e-7, xtol=1e-7, gtol=1e-6)
    return rotations(result.x), {"success": bool(result.success), "nfev": result.nfev,
                                 "cost": float(result.cost), "message": result.message,
                                 "optimality": float(result.optimality),
                                 "fixed_gauge_frames": sorted(anchors)}


def _flatten(pairs):
    return (np.concatenate([np.full(len(a), i, int) for i, _, a, _ in pairs]),
            np.concatenate([np.full(len(a), j, int) for _, j, a, _ in pairs]),
            np.concatenate([a for _, _, a, _ in pairs]), np.concatenate([b for _, _, _, b in pairs]))


def _stats(errors, valid):
    errors = np.asarray(errors)
    # Invalid rays are included as a large error; the scored denominator is
    # fixed across candidates, unlike candidate-dependent visibility masking.
    values = np.where(valid & np.isfinite(errors), np.minimum(errors, 1e6), 2. * W)
    return {"matches": len(values), "invalid": int((~valid).sum()),
            "median_px": float(np.median(values)) if len(values) else None,
            "p90_px": float(np.percentile(values, 90)) if len(values) else None,
            "p95_px": float(np.percentile(values, 95)) if len(values) else None,
            "within_5px_share": float((values <= 5).mean()) if len(values) else None}


def score_profile(profile, poses, fit_pairs, score_pairs, max_nfev=120):
    rotations, solver = fit_rotations(poses, fit_pairs, profile, max_nfev)
    ii, jj, a, b = _flatten(score_pairs)
    ra, va = undistort_points(a, profile)
    rb, vb = undistort_points(b, profile)
    wa = np.einsum("nij,nj->ni", rotations[ii], ra)
    wb = np.einsum("nij,nj->ni", rotations[jj], rb)
    ab = np.einsum("nji,nj->ni", rotations[jj], wa)
    ba = np.einsum("nji,nj->ni", rotations[ii], wb)
    uvb, pab = project_bearings(ab, profile)
    uva, pba = project_bearings(ba, profile)
    errors = np.sqrt(.5 * (((uvb - b)**2).sum(axis=1) + ((uva - a)**2).sum(axis=1)))
    valid = va & vb & pab & pba
    border = np.maximum(np.max(np.abs((a - [W / 2, H / 2]) / [W / 2, H / 2]), axis=1),
                        np.max(np.abs((b - [W / 2, H / 2]) / [W / 2, H / 2]), axis=1))
    return {"all": _stats(errors, valid), "edge": _stats(errors[border >= .75], valid[border >= .75]),
            "centre": _stats(errors[border < .5], valid[border < .5]), "solver": solver,
            "fit_pairs": len(fit_pairs), "score_pairs": len(score_pairs),
            "fit_matches": sum(len(a) for _, _, a, _ in fit_pairs)}


def run(sweep, matches, profiles, out, training_sweep=None, seed=20261106, max_nfev=120):
    poses = json.loads((Path(sweep) / "poses.json").read_text())["poses"]
    pairs = load_matches(matches, poses, seed)
    fit_pairs, score_pairs = validation_split(pairs, len(poses), seed)
    overlap = capture_overlap(training_sweep, sweep) if training_sweep else None
    if overlap and not overlap["independent_captures"]:
        raise ValueError("training and validation sweeps share captured image hashes")
    membership = {"fit_pairs": [(i, j) for i, j, _, _ in fit_pairs],
                  "score_pairs": [(i, j) for i, j, _, _ in score_pairs]}
    evidence = {"method": "fixed intrinsics; nuisance rotations fit on disjoint pairs AND feature coordinates; symmetric pixel score",
                "orientation_fit": "soft_l1 spherical bearing residuals, scale3px; data-only robust spanning-forest initialization",
                "invalid_prediction_policy": "behind-camera/nonfinite/past-first-radial-turning-point predictions count3840px; fixed denominator",
                "validator_sha256": _digest(__file__),
                "feature_assignment": "seeded blake2b of frame index and coordinates rounded to0.01px; both endpoints must agree",
                "edge_definition": "either endpoint beyond 75% of the horizontal or vertical image half-size",
                "limitations": ["natural-scene matches are not known-target truth", "rotation-only model excludes pivot translation/parallax",
                                "homography RANSAC is a shared raw-image prefilter and can favour homographic image regions"],
                "seed": seed, "sweep": str(sweep), "poses_sha256": _digest(Path(sweep) / "poses.json"),
                "matches_sha256": _digest(matches), "capture_overlap": overlap,
                "training_sweep": str(training_sweep) if training_sweep else None,
                "split_sha256": hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest(),
                **membership, "profiles": {}}
    for name, path in profiles.items():
        profile = json.loads(Path(path).read_text())
        result = score_profile(profile, poses, fit_pairs, score_pairs, max_nfev)
        evidence["profiles"][name] = {"path": str(path), "sha256": _digest(path),
                                      "intrinsics": {k: profile[k] for k in ("focal_px", "k1", "k2")}, **result}
        print(name, json.dumps(result), flush=True)
        # Preserve completed results even when a later candidate is interrupted.
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--profile", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--training-sweep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--max-nfev", type=int, default=120)
    a = ap.parse_args()
    run(a.sweep, a.matches, dict(p.split("=", 1) for p in a.profile), a.out, a.training_sweep, a.seed, a.max_nfev)


if __name__ == "__main__":
    main()
